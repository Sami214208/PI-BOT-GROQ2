import cv2
import os
import time
import base64
from groq import Groq

from dotenv import load_dotenv
load_dotenv()


class VisionService:
    def __init__(self, default_language="Hungarian"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.default_language = default_language

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

        
    def get_whats_visible_on_camera(self):
        file_path = os.path.abspath(os.path.dirname(__file__)) + "/"
        file_name = 'capture.png'      
        local_file = os.path.join(file_path, file_name)
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)        
        result, image = cam.read()
        cam.release()
        if not result:
            raise RuntimeError("Could not capture an image from the camera")
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if not cv2.imwrite(local_file, rotated):
            raise RuntimeError(f"Could not save camera image to {local_file}")
        base64_image = self.encode_image(local_file)     

        response = self.client.chat.completions.create(
            model=os.getenv('GROQ_VISION_MODEL') or 'qwen/qwen3.8-27b',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 
                        f'''Act as an AI assistant device who has vision. Describe what you can see. Respond in {self.default_language}.''' },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                            }
                        }                              
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def checkcamera(self):
        print('Checking Device...... \n')
        ret = os.popen("ls /dev/video*").read()
        video_list = []
        for i in ret.split('\n'):
            video_list.append((i.split('/')[-1][5:]))
        for i in video_list[:-1]:
            if int(i) < 10:
                print('Current device: video' + i + '\n')
                return int(i)
        return None    
            
if __name__ == "__main__":
    vision_service = VisionService()
    print(vision_service.get_whats_visible_on_camera())