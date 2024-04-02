/*
 * cvapp.cpp
 *
 *  Created on: 2018�~12��4��
 *      Author: 902452
 */

#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp_yolov5n_ob.h"
#include "cisdp_sensor.h"

#include "WE2_core.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "img_proc_helium.h"
#include "yolo_postprocessing.h"


#include "xprintf.h"
#include "spi_master_protocol.h"
#include "cisdp_cfg.h"
#include "memory_manage.h"
#include <send_result.h>

#define CHANGE_YOLOV5_OB_OUPUT_SHAPE 0


#define INPUT_IMAGE_CHANNELS 3

#if 1
#define YOLOV5_OB_INPUT_TENSOR_WIDTH   160
#define YOLOV5_OB_INPUT_TENSOR_HEIGHT  160
#define YOLOV5_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS
#else
#define YOLOV5_OB_INPUT_TENSOR_WIDTH   160
#define YOLOV5_OB_INPUT_TENSOR_HEIGHT  160
#define YOLOV5_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS
#endif

#define YOLOV5N_OB_DBG_APP_LOG 0


// #define EACH_STEP_TICK
#define TOTAL_STEP_TICK
#define YOLOV5_POST_EACH_STEP_TICK 0
uint32_t systick_1, systick_2;
uint32_t loop_cnt_1, loop_cnt_2;
#define CPU_CLK	0xffffff+1
static uint32_t capture_image_tick = 0;
#ifdef TRUSTZONE_SEC
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE	BASE_ADDR_APB_U55_CTRL
#endif
#endif


using namespace std;

namespace {

constexpr int tensor_arena_size = 1053*1024;

static uint32_t tensor_arena=0;

struct ethosu_driver ethosu_drv; /* Default Ethos-U device driver */
tflite::MicroInterpreter *yolov5n_ob_int_ptr=nullptr;
TfLiteTensor *yolov5n_ob_input, *yolov5n_ob_output;
};

#if YOLOV5N_OB_DBG_APP_LOG
std::string coco_classes[] = {"airpods","mario"};
int coco_ids[] = {0, 1};

#endif

static void _arm_npu_irq_handler(void)
{
    /* Call the default interrupt handler from the NPU driver */
    ethosu_irq_handler(&ethosu_drv);
}

/**
 * @brief  Initialises the NPU IRQ
 **/
static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;

    /* Register the EthosU IRQ handler in our vector table.
     * Note, this handler comes from the EthosU driver */
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);

    /* Enable the IRQ */
    NVIC_EnableIRQ(ethosu_irqnum);

}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    /* Initialise the IRQ */
    _arm_npu_irq_init();

    /* Initialise Ethos-U55 device */
#if TFLM2209_U55TAG2205
	const void * ethosu_base_address = (void *)(U55_BASE);
#else 
	void * const ethosu_base_address = (void *)(U55_BASE);
#endif

    if (0 != (err = ethosu_init(
                            &ethosu_drv,             /* Ethos-U driver device pointer */
                            ethosu_base_address,     /* Ethos-U NPU's base address. */
                            NULL,       /* Pointer to fast mem area - NULL for U55. */
                            0, /* Fast mem region size. */
							security_enable,                       /* Security enable. */
							privilege_enable))) {                   /* Privilege enable. */
    	xprintf("failed to initalise Ethos-U device\n");
            return err;
        }

    xprintf("Ethos-U55 device initialised\n");

    return 0;
}


int cv_yolov5n_ob_init(bool security_enable, bool privilege_enable, uint32_t model_addr) {
	int ercode = 0;

	//set memory allocation to tensor_arena
	tensor_arena = mm_reserve_align(tensor_arena_size,0x20); //1mb
	xprintf("TA[%x]\r\n",tensor_arena);


	if(_arm_npu_init(security_enable, privilege_enable)!=0)
		return -1;

	if(model_addr != 0) {
		static const tflite::Model*yolov5n_ob_model = tflite::GetModel((const void *)model_addr);

		if (yolov5n_ob_model->version() != TFLITE_SCHEMA_VERSION) {
			xprintf(
				"[ERROR] YOLOV5n_ob_model's schema version %d is not equal "
				"to supported version %d\n",
				yolov5n_ob_model->version(), TFLITE_SCHEMA_VERSION);
			return -1;
		}
		else {
			xprintf("yolov5n_ob model's schema version %d\n", yolov5n_ob_model->version());
		}

		static tflite::MicroErrorReporter yolov5n_ob_micro_error_reporter;
		static tflite::MicroMutableOpResolver<2> yolov5n_ob_op_resolver;

		yolov5n_ob_op_resolver.AddTranspose();
		if (kTfLiteOk != yolov5n_ob_op_resolver.AddEthosU()){
			xprintf("Failed to add Arm NPU support to op resolver.");
			return false;
		}
		#if TFLM2209_U55TAG2205
			static tflite::MicroInterpreter yolov5n_ob_static_interpreter(yolov5n_ob_model, yolov5n_ob_op_resolver,
					(uint8_t*)tensor_arena, tensor_arena_size, &yolov5n_ob_micro_error_reporter);
		#else
			static tflite::MicroInterpreter yolov5n_ob_static_interpreter(yolov5n_ob_model, yolov5n_ob_op_resolver,
					(uint8_t*)tensor_arena, tensor_arena_size);  
		#endif  


		if(yolov5n_ob_static_interpreter.AllocateTensors()!= kTfLiteOk) {
			return false;
		}
		yolov5n_ob_int_ptr = &yolov5n_ob_static_interpreter;
		yolov5n_ob_input = yolov5n_ob_static_interpreter.input(0);
		yolov5n_ob_output = yolov5n_ob_static_interpreter.output(0);
		#if CHANGE_YOLOV5_OB_OUPUT_SHAPE
			yolov5n_ob_output2 = YOLOV5n_ob_static_interpreter.output(1);
		#endif
	}

	xprintf("initial done\n");
	return ercode;
}



typedef struct detection_cls_yolov5{
    box bbox;
    float confidence;
    float index;

} detection_cls_yolov5;

static bool yolov5_det_comparator(detection_cls_yolov5 &pa, detection_cls_yolov5 &pb)
{
    return pa.confidence > pb.confidence;
}


static void  yolov5_NMSBoxes(std::vector<box> &boxes,std::vector<float> &confidences,float modelScoreThreshold,float modelNMSThreshold,std::vector<int>& nms_result)
{
    detection_cls_yolov5 yolov5_bbox;
    std::vector<detection_cls_yolov5> yolov5_bboxes{};
    for(int i = 0; i < boxes.size(); i++)
    {
        yolov5_bbox.bbox = boxes[i];
        yolov5_bbox.confidence = confidences[i];
        yolov5_bbox.index = i;
        yolov5_bboxes.push_back(yolov5_bbox);
    }
    sort(yolov5_bboxes.begin(), yolov5_bboxes.end(), yolov5_det_comparator);
    int updated_size = yolov5_bboxes.size();
    for(int k = 0; k < updated_size; k++)
    {
        if(yolov5_bboxes[k].confidence < modelScoreThreshold)
        {
            continue;
        }
        
        nms_result.push_back(yolov5_bboxes[k].index);
        for(int j = k + 1; j < updated_size; j++)
        {
            float iou = box_iou(yolov5_bboxes[k].bbox, yolov5_bboxes[j].bbox);
            // float iou = box_diou(yolov5_bboxes[k].bbox, yolov5_bboxes[j].bbox);
            if(iou > modelNMSThreshold)
            {
                yolov5_bboxes.erase(yolov5_bboxes.begin() + j);
                updated_size = yolov5_bboxes.size();
                j = j -1;
            }
        }

    }
}



#if CHANGE_YOLOV5_OB_OUPUT_SHAPE
static void YOLOV5_ob_post_processing(tflite::MicroInterpreter* static_interpreter,float modelScoreThreshold, float modelNMSThreshold, struct_yolov5_ob_algoResult *alg,	std::forward_list<el_box_t> &el_algo)
{
	uint32_t img_w = app_get_raw_width();
    uint32_t img_h = app_get_raw_height();
	TfLiteTensor* output = static_interpreter->output(0);
	TfLiteTensor* output_2 = static_interpreter->output(1);
	// init postprocessing 	
	int num_classes = output_2->dims->data[2];

	#if YOLOV5N_OB_DBG_APP_LOG
		xprintf("output->dims->data[0]: %d\r\n",output->dims->data[0]);//1
		xprintf("output->dims->data[1]: %d\r\n",output->dims->data[1]);//4
		xprintf("output->dims->data[2]: %d\r\n",output->dims->data[2]);//756

		xprintf("output_2->dims->data[0]: %d\r\n",output_2->dims->data[0]);//1
		xprintf("output_2->dims->data[1]: %d\r\n",output_2->dims->data[1]);//756
		xprintf("output_2->dims->data[2]: %d\r\n",output_2->dims->data[2]);//80
	#endif
	// end init
	///////////////////////
	// start postprocessing
	int nboxes=0;
	int input_w = YOLOV5_OB_INPUT_TENSOR_WIDTH;
	int input_h = YOLOV5_OB_INPUT_TENSOR_HEIGHT;

	std::vector<uint16_t> class_idxs;
	std::vector<float> confidences;
	std::vector<box> boxes;


	float output_scale = ((TfLiteAffineQuantization*)(output->quantization.params))->scale->data[0];
	int output_zeropoint = ((TfLiteAffineQuantization*)(output->quantization.params))->zero_point->data[0];

	float output_2_scale = ((TfLiteAffineQuantization*)(output_2->quantization.params))->scale->data[0];
	int output_2_zeropoint = ((TfLiteAffineQuantization*)(output_2->quantization.params))->zero_point->data[0];


	#if YOLOV5N_OB_DBG_APP_LOG
		printf("output_scale: %f\r\n",output_scale);
		xprintf("output_zeropoint: %d\r\n",output_zeropoint);

		printf("output_2_scale: %f\r\n",output_2_scale);
		xprintf("output_2_zeropoint: %d\r\n",output_2_zeropoint);
	#endif
	/***
	 * dequantize the output result for box
	 * 
	 * 
	 ******/
	for(int dims_cnt_2 = 0; dims_cnt_2 < output->dims->data[2]; dims_cnt_2++)//// output->dims->data[2] is 756
	{
		float outputs_bbox_data[4];
		float maxScore = (-1);// the first four indexes are bbox information
		uint16_t maxClassIndex = 0;
		for(int dims_cnt_1 = 0; dims_cnt_1 < output->dims->data[1]; dims_cnt_1++)// output->dims->data[1] is 4 
		{
			int value =  output->data.int8[ dims_cnt_2 + dims_cnt_1 * output->dims->data[2]];
			
			float deq_value = ((float) value-(float)output_zeropoint) * output_scale ;

			/***
			 * fix big score
			 * ****/
			if(dims_cnt_1%2)//==1
			{
				deq_value *= (float)input_h;
			}
			else
			{
				deq_value *= (float)input_w;
			}
			outputs_bbox_data[dims_cnt_1] = deq_value;
		}

		for(int output_2_dims_cnt_1 = 0; output_2_dims_cnt_1 < output_2->dims->data[2]; output_2_dims_cnt_1++)//output_2->dims->data[2] is 80
		{
			int value_2 =  output_2->data.int8[ output_2_dims_cnt_1 + dims_cnt_2 * output_2->dims->data[2]];
			
			float deq_value_2 = ((float) value_2-(float)output_2_zeropoint) * output_2_scale ;
			/***
			 * find maximum Score and correspond Class idx
			 * **/
			if(maxScore < deq_value_2)
			{
				maxScore = deq_value_2;
				maxClassIndex = output_2_dims_cnt_1;
			}
		}
		if (maxScore >= modelScoreThreshold)
		{
			box bbox;
			
			bbox.x = (outputs_bbox_data[0] - (0.5 * outputs_bbox_data[2]));
			bbox.y = (outputs_bbox_data[1] - (0.5 * outputs_bbox_data[3]));
			bbox.w =(outputs_bbox_data[2]);
			bbox.h = (outputs_bbox_data[3]);
			boxes.push_back(bbox);
			class_idxs.push_back(maxClassIndex);
			confidences.push_back(maxScore);
			
		}
	}

	
	#if YOLOV5N_OB_DBG_APP_LOG
		xprintf("boxes.size(): %d\r\n",boxes.size());
	#endif
	/**
	 * do nms
	 * 
	 * **/

	std::vector<int> nms_result;
	YOLOV5_NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
	#if YOLOV5N_OB_DBG_APP_LOG
		xprintf("nms_result.size(): %d\r\n",nms_result.size());
	#endif
	for (int i = 0; i < nms_result.size(); i++)
	{
		if(!(MAX_TRACKED_YOLOV5_ALGO_RES-i))break;
		int idx = nms_result[i];

		float scale_factor_w = (float)img_w / (float)YOLOV5_OB_INPUT_TENSOR_WIDTH; 
		float scale_factor_h = (float)img_h / (float)YOLOV5_OB_INPUT_TENSOR_HEIGHT; 
		alg->obr[i].confidence = confidences[idx];
		alg->obr[i].bbox.x = (uint32_t)(boxes[idx].x * scale_factor_w);
		alg->obr[i].bbox.y = (uint32_t)(boxes[idx].y * scale_factor_h);
		alg->obr[i].bbox.width = (uint32_t)(boxes[idx].w * scale_factor_w);
		alg->obr[i].bbox.height = (uint32_t)(boxes[idx].h * scale_factor_h);
		alg->obr[i].class_idx = class_idxs[idx];
		el_box_t temp_el_box;
		temp_el_box.score =  confidences[idx]*100;
		temp_el_box.target =  class_idxs[idx];
		temp_el_box.x = (uint32_t)(boxes[idx].x * scale_factor_w);
		temp_el_box.y =  (uint32_t)(boxes[idx].y * scale_factor_h);
		temp_el_box.w = (uint32_t)(boxes[idx].w * scale_factor_w);
		temp_el_box.h = (uint32_t)(boxes[idx].h * scale_factor_h);


		// printf("temp_el_box.x %d,temp_el_box.y: %d\r\n",temp_el_box.x,temp_el_box.y);
		el_algo.emplace_front(temp_el_box);
		// for (auto box : el_algo) {
		// 	printf("el_algo.box.x %d,el_algo.box.y%d\r\n",box.x,box.y);
		// }
		#if YOLOV5N_OB_DBG_APP_LOG
			printf("detect object[%d]: %s confidences: %f\r\n",i, coco_classes[class_idxs[idx]].c_str(),confidences[idx]);

		#endif
	}
}
#else
static void YOLOV5_ob_post_processing(tflite::MicroInterpreter* static_interpreter,float modelScoreThreshold, float modelNMSThreshold, struct_yolov5_ob_algoResult *alg,	std::forward_list<el_box_t> &el_algo)
{
	uint32_t img_w = app_get_raw_width();
    uint32_t img_h = app_get_raw_height();
	TfLiteTensor* output = static_interpreter->output(0);
	// init postprocessing 	
	
	printf("Output dimensions[0] are: %d\r\n",output->dims->data[0]);
	printf("Output dimensions[1] are: %d\r\n",output->dims->data[1]);
	printf("Output dimensions[2] are: %d\r\n",output->dims->data[2]);
	int num_classes = output->dims->data[2] - 5;

	
	// end init
	///////////////////////
	// start postprocessing
	int nboxes=0;
	int input_w = YOLOV5_OB_INPUT_TENSOR_WIDTH;
	int input_h = YOLOV5_OB_INPUT_TENSOR_HEIGHT;

	std::vector<uint16_t> class_idxs;
	std::vector<float> confidences;
	std::vector<box> boxes;


	float output_scale = ((TfLiteAffineQuantization*)(output->quantization.params))->scale->data[0];
	int output_zeropoint = ((TfLiteAffineQuantization*)(output->quantization.params))->zero_point->data[0];
	int output_size = output->bytes;

	#if YOLOV5N_OB_DBG_APP_LOG
		// xprintf("output->dims->size: %d\r\n",output->dims->size);
		// printf("output_scale: %f\r\n",output_scale);
		// xprintf("output_zeropoint: %d\r\n",output_zeropoint);
		// xprintf("output_size: %d\r\n",output_size);
		// xprintf("output->dims->data[0]: %d\r\n",output->dims->data[0]);//1
		// xprintf("output->dims->data[1]: %d\r\n",output->dims->data[1]);//84
		// xprintf("output->dims->data[2]: %d\r\n",output->dims->data[2]);//756
	#endif
	/***
	 * dequantize the output result
	 * 
	 * 
	 ******/
	for(int dims_cnt_2 = 0; dims_cnt_2 < output->dims->data[1]; dims_cnt_2++)
	{
		float outputs_bbox_data[4];
		float maxScore = (-1);// the first four indexes are bbox information
		float objectness = (0.0);
		uint16_t maxClassIndex = 0;
		for(int dims_cnt_1 = 0; dims_cnt_1 < output->dims->data[2]; dims_cnt_1++)
		{
			int value =  output->data.int8[ dims_cnt_1 + dims_cnt_2 * output->dims->data[2]];
			
			float deq_value = ((float) value-(float)output_zeropoint) * output_scale ;
			if(dims_cnt_1<4)
			{
				/***
				 * fix big score
				 * ****/
				if(dims_cnt_1%2)//==1
				{
					deq_value *= (float)input_h;
				}
				else
				{
					deq_value *= (float)input_w;
				}
				outputs_bbox_data[dims_cnt_1] = deq_value;
			}
			else if(dims_cnt_1 > 4)
			{
				/***
				 * find maximum Score and correspond Class idx
				 * **/
				if(maxScore < deq_value)
				{
					maxScore = deq_value;
					maxClassIndex = dims_cnt_1-4;
				}
			}
			else if(dims_cnt_1 == 4){
				objectness = deq_value;
			}

		}
		if (objectness >= modelScoreThreshold){
			if (maxScore >= modelScoreThreshold)
				{
					box bbox;
					//this code makes the xcenter and ycenter of the bbox to be at the top left corner of the cell
					//CORRECT!
					bbox.x = (outputs_bbox_data[0] - (0.5 * outputs_bbox_data[2]));
					bbox.y = (outputs_bbox_data[1] - (0.5 * outputs_bbox_data[3]));
					// bbox.x = (outputs_bbox_data[0]);
					// bbox.y = (outputs_bbox_data[1]);
					bbox.w =(outputs_bbox_data[2]);
					bbox.h = (outputs_bbox_data[3]);
					boxes.push_back(bbox);
					class_idxs.push_back(maxClassIndex);
					confidences.push_back(maxScore);
					
				}
			}
	}
	#if YOLOV5N_OB_DBG_APP_LOG
		xprintf("boxes.size(): %d\r\n",boxes.size());
	#endif
	/**
	 * do nms
	 * 
	 * **/

	std::vector<int> nms_result;
	yolov5_NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++)
	{
		if(!(MAX_TRACKED_YOLOV5_ALGO_RES-i))break;
		int idx = nms_result[i];

		float scale_factor_w = (float)img_w / (float)YOLOV5_OB_INPUT_TENSOR_WIDTH; 
		float scale_factor_h = (float)img_h / (float)YOLOV5_OB_INPUT_TENSOR_HEIGHT; 
		alg->obr[i].confidence = confidences[idx];
		alg->obr[i].bbox.x = (uint32_t)(boxes[idx].x * scale_factor_w);
		alg->obr[i].bbox.y = (uint32_t)(boxes[idx].y * scale_factor_h);
		alg->obr[i].bbox.width = (uint32_t)(boxes[idx].w * scale_factor_w);
		alg->obr[i].bbox.height = (uint32_t)(boxes[idx].h * scale_factor_h);
		alg->obr[i].class_idx = class_idxs[idx];

		el_box_t temp_el_box;
		temp_el_box.score =  confidences[idx]*100;
		temp_el_box.target =  class_idxs[idx];
		temp_el_box.x = (uint32_t)(boxes[idx].x * scale_factor_w);
		temp_el_box.y =  (uint32_t)(boxes[idx].y * scale_factor_h);
		temp_el_box.w = (uint32_t)(boxes[idx].w * scale_factor_w);
		temp_el_box.h = (uint32_t)(boxes[idx].h * scale_factor_h);

		el_algo.emplace_front(temp_el_box);

		#if YOLOV5N_OB_DBG_APP_LOG
			printf("detect object[%d]: %s confidences: %f\r\n",i, coco_classes[class_idxs[idx]].c_str(),confidences[idx]);

		#endif
	}
}

#endif

int cv_yolov5n_ob_run(struct_yolov5_ob_algoResult *algoresult_yolov5n_ob) {
	int ercode = 0;
    float w_scale;
    float h_scale;
    uint32_t img_w = app_get_raw_width();
    uint32_t img_h = app_get_raw_height();
    uint32_t ch = app_get_raw_channels();
	//this is the flash memory address
    uint32_t raw_addr = app_get_raw_addr();
    uint32_t expand = 0;
	std::forward_list<el_box_t> el_algo;

	#if YOLOV5N_OB_DBG_APP_LOG
    xprintf("raw info: w[%d] h[%d] ch[%d] addr[%x]\n",img_w, img_h, ch, raw_addr);
	#endif

    if(yolov5n_ob_int_ptr!= nullptr) {
		#ifdef TOTAL_STEP_TICK
			SystemGetTick(&systick_1, &loop_cnt_1);
		#endif
		#ifdef EACH_STEP_TICK
			SystemGetTick(&systick_1, &loop_cnt_1);
		#endif
    	//get image from sensor and resize
		w_scale = (float)(img_w - 1) / (YOLOV5_OB_INPUT_TENSOR_WIDTH - 1);
		h_scale = (float)(img_h - 1) / (YOLOV5_OB_INPUT_TENSOR_HEIGHT - 1);

		
		hx_lib_image_resize_BGR8U3C_to_RGB24_helium((uint8_t*)raw_addr, (uint8_t*)yolov5n_ob_input->data.data,  
		                    img_w, img_h, ch, 
                        	YOLOV5_OB_INPUT_TENSOR_WIDTH, YOLOV5_OB_INPUT_TENSOR_HEIGHT, w_scale,h_scale);
		#ifdef EACH_STEP_TICK						
			SystemGetTick(&systick_2, &loop_cnt_2);
			dbg_printf(DBG_LESS_INFO,"Tick for resize image BGR8U3C_to_RGB24_helium for YOLOV5 OB:[%d]\r\n",(loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2));							
		#endif

		#ifdef EACH_STEP_TICK
			SystemGetTick(&systick_1, &loop_cnt_1);
		#endif

		// //uint8 to int8
		for (int i = 0; i < yolov5n_ob_input->bytes; ++i) {
			*((int8_t *)yolov5n_ob_input->data.data+i) = *((int8_t *)yolov5n_ob_input->data.data+i) - 128;
    	}

		#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_2, &loop_cnt_2);
		dbg_printf(DBG_LESS_INFO,"Tick for Invoke for uint8toint8 for YOLOV5_OB:[%d]\r\n\n",(loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2));    
		#endif	

		#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_1, &loop_cnt_1);
		#endif
		TfLiteStatus invoke_status = yolov5n_ob_int_ptr->Invoke();

		#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_2, &loop_cnt_2);
		#endif
		if(invoke_status != kTfLiteOk)
		{
			xprintf("YOLOV5 object detect invoke fail\n");
			return -1;
		}
		else
		{
			#if YOLOV5N_OB_DBG_APP_LOG
			xprintf("YOLOV5 object detect  invoke pass\n");
			#endif
		}
		#ifdef EACH_STEP_TICK
    		dbg_printf(DBG_LESS_INFO,"Tick for Invoke for YOLOV5_OB:[%d]\r\n\n",(loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2));    
		#endif

		#ifdef EACH_STEP_TICK
			SystemGetTick(&systick_1, &loop_cnt_1);
		#endif
		//retrieve output data
		YOLOV5_ob_post_processing(yolov5n_ob_int_ptr,0.5, 0.45, algoresult_yolov5n_ob,el_algo);
		#ifdef EACH_STEP_TICK
			SystemGetTick(&systick_2, &loop_cnt_2);
			dbg_printf(DBG_LESS_INFO,"Tick for Invoke for YOLOV5_OB_post_processing:[%d]\r\n\n",(loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2));    
		#endif
		#if YOLOV5N_OB_DBG_APP_LOG
			xprintf("YOLOV5_ob_post_processing done\r\n");
		#endif
		#ifdef TOTAL_STEP_TICK						
			SystemGetTick(&systick_2, &loop_cnt_2);
			// dbg_printf(DBG_LESS_INFO,"Tick for TOTAL YOLOV5 OB:[%d]\r\n",(loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2));		
		#endif

    }
	

#ifdef UART_SEND_ALOGO_RESEULT
	algoresult_yolov5n_ob->algo_tick = (loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2) + capture_image_tick;
uint32_t judge_case_data;
uint32_t g_trans_type;
hx_drv_swreg_aon_get_appused1(&judge_case_data);
g_trans_type = (judge_case_data>>16);
if( g_trans_type == 0 || g_trans_type == 2)// transfer type is (UART) or (UART & SPI) 
{
	el_img_t temp_el_jpg_img = el_img_t{};
	temp_el_jpg_img.data = (uint8_t *)app_get_jpeg_addr();
	temp_el_jpg_img.size = app_get_jpeg_sz();
	temp_el_jpg_img.width = app_get_raw_width();
	temp_el_jpg_img.height = app_get_raw_height();
	temp_el_jpg_img.format = EL_PIXEL_FORMAT_JPEG;
	temp_el_jpg_img.rotate = EL_PIXEL_ROTATE_0;

	send_device_id();
	// event_reply(concat_strings(", ", box_results_2_json_str(el_algo), ", ", img_2_json_str(&temp_el_jpg_img)));
	event_reply(concat_strings(", ", algo_tick_2_json_str(algoresult_yolov5n_ob->algo_tick),", ", box_results_2_json_str(el_algo), ", ", img_2_json_str(&temp_el_jpg_img)));
}
	set_model_change_by_uart();
#endif	

	SystemGetTick(&systick_1, &loop_cnt_1);
	//recapture image
	sensordplib_retrigger_capture();

	
	SystemGetTick(&systick_2, &loop_cnt_2);
	capture_image_tick = (loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2);	
	return ercode;
}

int cv_yolov5n_ob_deinit()
{
	
	return 0;
}

