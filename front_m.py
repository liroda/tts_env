#!/usr/bin/env  python3
import os,sys,time,re,json
import numpy as np
from vits_textfront import VITS_TextFront
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):

        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
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

        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "x")
        output1_config = pb_utils.get_output_config_by_name(model_config, "x_length")
        output2_config = pb_utils.get_output_config_by_name(model_config, "word_input_ids")
        output3_config = pb_utils.get_output_config_by_name(model_config, "expand_length")

         # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])
        self.output2_dtype = pb_utils.triton_string_to_numpy(output2_config['data_type'])
        self.output3_dtype = pb_utils.triton_string_to_numpy(output3_config['data_type'])


        front_dict="/models/tts_front_poly/dict"
        phoneid_map_path= os.path.join(front_dict,"mix_phone_id_map.txt")
        jieba_userdict_path = os.path.join(front_dict,"jieba_userdict.txt")
        poly_dict_path = os.path.join(front_dict,"polypinyin.txt")
        bert_dict_path = os.path.join(front_dict,"vocab.txt")
        symbol_pinyin_path = os.path.join(front_dict,"pinyin_symbol.txt")
        pinyin2phone_path = os.path.join(front_dict,"pinyin_phones.txt")
        cmudict_path = os.path.join(front_dict,"cmu_dict.txt")
        polyword_dict_path = os.path.join(front_dict,"polyword.list")
        polymodel_dir="/models/tts_front_poly/1/poly_g2pw"

        self.frontend = VITS_TextFront(
                                 phoneid_map_path =phoneid_map_path,
                                 jieba_userdict_path=jieba_userdict_path,
                                 bert_dict_path = bert_dict_path,
                                 poly_dict_path=poly_dict_path,
                                 symbol_pinyin_path = symbol_pinyin_path ,
                                 pinyin2phone_path = pinyin2phone_path,
                                 cmudict_path = cmudict_path,
                                 polyword_dict_path = polyword_dict_path,
                                 polymodel_dir = polymodel_dir)


    def execute(self,requests):

        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
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
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype
        output3_dtype = self.output3_dtype

        responses = []
        for request in requests:
            input_text = pb_utils.get_input_tensor_by_name(request,"textstring")
            byte_text_list = input_text.as_numpy().tolist()


            textstr = byte_text_list[0].decode("utf-8")
            front_out = self.frontend.get_front_inputids_embeds(textstr)
            front_result = front_out[0]
            #print (front_result)
            

            x = np.array(front_result["x"]).astype(output0_dtype)
            x_length = np.array(front_result["x_length"]).astype(output1_dtype)
            word_input_ids = np.array(front_result["word_input_ids"]).astype(output2_dtype)
            expand_length = np.array(front_result["expand_length"]).astype(output3_dtype)
            #print (x,x_length,word_input_ids,expand_length)

            out0_tensor = pb_utils.Tensor("x",x)
            out1_tensor = pb_utils.Tensor("x_length",x_length)
            out2_tensor = pb_utils.Tensor("word_input_ids",word_input_ids)
            out3_tensor = pb_utils.Tensor("expand_length",expand_length)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0_tensor,out1_tensor,out2_tensor,out3_tensor])
            responses.append(inference_response)
        return responses 
    def finalize(self):
        print('Cleaning up...')
