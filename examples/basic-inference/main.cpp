/**
 * @author Todd Rylaarsdam <todd@toddr.org>
 * @date 2023-06-07
 * @brief Basic inference example using an LSTM based model. This example was 
 * created for the Portenta H7, but if you use a smaller model it can be used
 * anywhere.
*/

/**
 * The model we are including - this can be built by exporting a model to the .tflite
 * file format and then using `xxd -i model.tflite > model.h` to generate the header file
*/
#include "lstm-model.h"

/**
 * The includes for Tensorflow Lite for Microcontrollers.
 * You'll notice that there are 2 different headers available for the ops resolver:
 * either the all_ops or micro_mutable_op resolver. The all_ops resolver is great
 * for development when your model can be changing and such, however it is best to
 * swtich to the micro_mutable_op_resolver when your model's operations are set.
 * 
 * This example uses the all ops resolver, but I'm including commented out implementations
 * of the micro_mutable_op_resolver for reference. To see what ops your model uses,
 * you can use https://netron.app to view your model structure
*/
#include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h" 
// #define NUM_TF_OPS 5 // this is the number of ops in the model, used by the micro_mutable_op_resolver

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
/**
 * @note The micro logger by default prints to stdout, which is not
 * the same as the Portenta H7's USB CDC serial port. To enable logging
 * with the serial port you can go to this file in the library
 * `src/tensorflow/lite/micro/micro_log.cc` and add the following:
 * *Below all the includes*, but above the Log function, add:
 * #include "Arduino.h"
 * Inside the void Log(const char* format, va_list args) function: add
 * Serial.println(log_buffer);
 * after line 35
*/
#include "tensorflow/lite/micro/micro_log.h"

/**
 * Important that the Arduino include comes last if on the Arduino platform, as it has an `abs()` function
 * that will screw with the stdlib abs() function. If need you can use the following lines
 * as well to redeclare the abs() function to be compatible
*/
#include "Arduino.h"
#ifdef ARDUINO
#define abs(x) ((x)>0?(x):-(x))
#endif 

/**
 * The size of the memory arena to use for the model's tensors. You'll
 * need to play with this value to make it suitable for your model. Start 
 * out with a large value, and then check the 
*/
#define RAM_SIZE 400000 // 400000 bytes - this is large because of this example model, but you can reduce this for smaller models
uint8_t tensor_arena[RAM_SIZE]; // where the model will be run

// Globally accessible interpreter
std::unique_ptr<tflite::MicroInterpreter> interpreter;

void setup() 
{
	// set up the error reporter
	static tflite::MicroErrorReporter micro_error_reporter;
	tflite::ErrorReporter* error_reporter = &micro_error_reporter;

	// set up the model
	const tflite::Model* model = tflite::GetModel(lstm_tflite);
	// check to make sure the model is compatible
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		#ifdef ARDUINO // serial out for Arduino instead of stdout
		Serial.print("Model provided is schema version ");
		Serial.print(model->version());
		Serial.print(" not equal to supported version ");
		Serial.println(TFLITE_SCHEMA_VERSION);
		#endif
		TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}

	// set up the all ops resolver
	static tflite::AllOpsResolver resolver;

	// set up the optional micro mutable ops resolver, and add needed operations
	// static tflite::MicroMutableOpResolver<NUM_TF_OPS> resolver;
	// resolver.AddUnidirectionalSequenceLSTM();
	// resolver.AddTanh();
	// resolver.AddFullyConnected();
	// resolver.AddStridedSlice();
	// resolver.AddLogistic();


	// Declare the TF lite interpreter
	static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, RAM_SIZE);
  interpreter = std::unique_ptr<tflite::MicroInterpreter>(&static_interpreter);

	// Allocate memory for the model's input buffers
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk)
	{
		TF_LITE_REPORT_ERROR(error_reporter, "Tensor allocation failed");
	}
	else
	{
		Serial.println("Tensor allocation success");
		Serial.print("Used bytes: ");
		Serial.println(interpreter->arena_used_bytes());
	}

	// Obtain a pointer to the model's input tensor
	TfLiteTensor *input = interpreter->input(0);

	// Print out the input tensor's details to verify
	// the model is working as expected
	Serial.print("Input size: ");
	Serial.println(input->dims->size);
	Serial.print("Input bytes: ");
	Serial.println(input->bytes);

	for (int i = 0; i < input->dims->size; i++)
	{
		Serial.print("Input dim ");
		Serial.print(i);
		Serial.print(": ");
		Serial.println(input->dims->data[i]);
	}

	// Supply data to the model
	// This model has a 1x3105x5 input, so the array needs to be 3105x5 floats long
	// The sequence of the array goes: {0:0, 0:1, 0:2, 0:3, 0:4, 1:0, 1:1, 1:2, 1:3, 1:4, ...}
	float input_data[3105 * 5] = {0};

	// Copy the data into the input tensor
	for (int i = 0; i < input->bytes; i++)
	{
		input->data.f[i] = input_data[i];
	}

	// Invoke the model
	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk)
	{
		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
	}
	else
	{
		Serial.println("Invoke completed");
	}

	// Obtain a pointer to the model's output tensor
	TfLiteTensor *output = interpreter->output(0);

	// Print out the output tensor's details to verify
	// the model is working as expected
	Serial.print("Output size: ");
	Serial.println(output->dims->size);
	Serial.print("Output bytes: ");
	Serial.println(output->bytes);

	for (int i = 0; i < output->dims->size; i++)
	{
		Serial.print("Output dim ");
		Serial.print(i);
		Serial.print(": ");
		Serial.println(output->dims->data[i]);
	}
}

void loop() 
{
	// nothing here for this example
}