import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class InferlessPythonModel:
    def initialize(self):

        model_id = "distil-whisper/distil-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
        model.to("cuda:0")
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline( "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, max_new_tokens=128, torch_dtype=torch.float16, device="cuda:0", )

    def infer(self,inputs):
        audio_url = inputs["audio_url"]
        pipeline_output = self.pipe(audio_url)
         
        return { "transcribed_output" : pipeline_output["text"]}
        
    def finalize(self):
        self.pipe = None
