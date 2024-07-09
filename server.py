import os,sys,re
from fastapi import APIRouter
from pydantic import BaseModel
import uvicorn
import traceback
from mylog import ttslogger
import tts_app


tts_front = tts_app.SFTTSFRONTEND()
tts_front.initfront()

app = APIRouter()

class Item(BaseModel):
    text: str
    request_id: str
    tag: bool=True

@app.get("/")
async def main():
    return {"message": "Hello，这是TTS 语音合成前端处理接口"}

@app.post("/tts_front")
async def run_tts_front(item: Item):

    # data check
    item_dict = item.dict()
    ttslogger.logger.info("request: {}".format(item_dict))

    text = item_dict["text"]
    request_id = item_dict["request_id"]
    tag = item_dict["tag"]

    
    result = {}
    try:
        data,time = tts_front.run(text)

        message = "success"
        code = 0
        result = {
                   "request_id": request_id,
                    "cost":time,
                    "data": data
                    }

    except Exception as e:
        message = traceback.format_exc()
        ttslogger.logger.error("Failed to synthesized audio {}".format(message))
        code = 1
        message = "ERROR"+message

 
    tts_result  = {"code": code,"msg": message}
    
    if result:
        tts_result.update(result)

    ttslogger.logger.info(tts_result)
    return tts_result

