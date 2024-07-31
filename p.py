import json
import numpy as np
import soxbindings as sox


# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "output0")
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])


    def postprocessor(self,v_out,volume_in,speed_in,target_fs=8000):
        # transform volume
        v_out_1 = np.expand_dims(v_out[0,:], axis=1)
        volume = volume_in[0]
        speed = speed_in[0]

        wav_vol = v_out_1 * volume
        # change_speed
        if speed != 1 :
            tfm = sox.Transformer()
            tfm.set_globals(multithread=False)
            tfm.tempo(speed)
            wav_speed = tfm.build_array(input_array=wav_vol,sample_rate_in=target_fs).astype(np.float32).copy()
            #wav_speed = wav_vol
        else:
            wav_speed = wav_vol
            
        # float32 转 int16
        dtype='int16'
        i = np.iinfo(dtype)
        abs_max = 2**(i.bits - 1)
        offset = i.min + abs_max
        wav_norm = (wav_speed * abs_max + offset).clip(i.min+1, i.max-1)
        wav_norm_int16 = wav_norm.astype(np.int16)

        return wav_norm_int16


    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "input0")
            in_volume = pb_utils.get_input_tensor_by_name(request, "volume")
            in_speed = pb_utils.get_input_tensor_by_name(request, "speed")

            out_0 = self.postprocessor(in_0.as_numpy(),in_volume.as_numpy(),in_speed.as_numpy(),8000)

            # Create output tensors. You need pb_utils.Tensor
            out_tensor_0 = pb_utils.Tensor("output0",out_0.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        print('Cleaning up...')
