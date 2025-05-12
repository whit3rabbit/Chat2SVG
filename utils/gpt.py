import os
import yaml
import base64
import mimetypes
import requests
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from utils.util import read

# Create a .env file in the project root directory and add your API keys
load_dotenv(dotenv_path=os.path.join("..", ".env"))


class Session:
    def __init__(self, model, prompts_file) -> None:
        self.model = model
        self.past_tasks: list[str] = []
        self.past_messages = []
        self.past_responses: list[str] = []

        # Load the predefined prompts for the LLM
        with open(f"{prompts_file}.yaml") as file:
            self.predefined_prompts: dict[str, str] = yaml.safe_load(file)

    def send(self, task: str, prompt_info: dict[str, str] | None = None, images: list[str] = [], file_path=None) -> str:
        print(f"$ --- Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info)
        self._send(prompt, images, file_path)
        response = self.past_responses[-1]
        print(f"$ --- Response:\n{response}\n")

        return response

    def _make_prompt(self, task: str, prompt_info: dict[str, str] | None) -> str:
        # Get the predefined prompt for the task
        prompt = self.predefined_prompts[task]

        # Check for task-specific required information ()
        # All tasks that require extra information should have a case here
        valid = True
        match task:
            case "expand_text_prompt":
                valid = "text_prompt" in prompt_info
        if not valid:
            raise ValueError(f"Extra information is required for the task: {task}")

        # Replace the placeholders in the prompt with the information
        if prompt_info is not None:
            for key in prompt_info:
                prompt = prompt.replace(f"<{key.upper()}>", prompt_info[key])

        return prompt

    def _send(self, prompt: str, images: list[str] = [], file_path=None) -> str:
        if not os.path.exists(file_path):
            # Try to get appropriate client based on model name
            if self.model.startswith("claude"):
                response = self._send_anthropic(prompt, images)
            elif self.model.startswith("gpt"):
                response = self._send_openai(prompt, images)
            else:
                # Default to the original implementation using WildCard
                payload = self._create_payload(prompt, images=images)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                }
                print("Waiting for LLM to be generated")
                # Use appropriate API depending on environment variables
                if os.getenv("ANTHROPIC_API_KEY"):
                    # Use direct Anthropic API
                    response = self._send_anthropic(prompt, images)
                elif os.getenv("OPENROUTER_API_KEY"):
                    # Use OpenRouter
                    response = self._send_openrouter(prompt, images)
                else:
                    # Fallback to WildCard
                    response = requests.post("https://api.gptsapi.net/v1/chat/completions", headers=headers, json=payload,
                                         timeout=(5, 120))  # WildCard
                    print(f"LLM response: {response.text}")
                    try:
                        response = response.json()['choices'][0]['message']['content']
                    except Exception as e:
                        print(f"$ --- Error Response: {response.json()}\n")
                        raise e
        else:
            response = read(file_path)
        
        self.past_messages.append({"role": "assistant", "content": response})
        self.past_responses.append(response)

    def _send_anthropic(self, prompt: str, images: list[str] = []) -> str:
        """Send request to Anthropic Claude API"""
        try:
            import anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
                
            client = anthropic.Anthropic(api_key=api_key)
            
            messages = []
            
            # Add image content if provided
            for image in images:
                with open(image, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")
                    mime_type, _ = mimetypes.guess_type(image)
                    if not mime_type:
                        mime_type = "image/png"  # Default to PNG if can't determine
                        
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data
                            }
                        }]
                    })
            
            # Add text content
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # For backwards compatibility
            if not messages and not images:
                messages = [{"role": "user", "content": prompt}]
                
            response = client.messages.create(
                model=self.model,
                system=self.predefined_prompts["system"],
                messages=messages,
                max_tokens=4096
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error sending to Anthropic: {str(e)}")
            raise

    def _send_openai(self, prompt: str, images: list[str] = []) -> str:
        """Send request to OpenAI API"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
                
            client = openai.OpenAI(api_key=api_key)
            
            messages = []
            
            # Add system message
            messages.append({
                "role": "system", 
                "content": self.predefined_prompts["system"]
            })
            
            # Add images if provided
            if images:
                content = []
                for image in images:
                    with open(image, "rb") as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                        mime_type, _ = mimetypes.guess_type(image)
                        if not mime_type:
                            mime_type = "image/png"
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}",
                                "detail": "high"
                            }
                        })
                
                content.append({
                    "type": "text",
                    "text": prompt
                })
                
                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                # No images, just text
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error sending to OpenAI: {str(e)}")
            raise

    def _send_openrouter(self, prompt: str, images: list[str] = []) -> str:
        """Send request to OpenRouter API"""
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
            # Use specified model or default to a known one on OpenRouter
            model = self.model
            if not model.startswith("openai/") and not model.startswith("anthropic/"):
                # Try to map to a known model on OpenRouter
                if model.startswith("gpt"):
                    model = f"openai/{model}"
                elif model.startswith("claude"):
                    model = f"anthropic/{model}"
            
            # Import OpenRouter or use requests directly
            try:
                from openrouter import OpenRouter
                client = OpenRouter(api_key=api_key)
                
                messages = []
                
                # Add system message
                system_message = self.predefined_prompts["system"]
                
                # Add images if provided (only for models that support them)
                if "gpt-4" in model or "claude" in model:
                    if images:
                        content = []
                        for image in images:
                            with open(image, "rb") as img_file:
                                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                                mime_type, _ = mimetypes.guess_type(image)
                                if not mime_type:
                                    mime_type = "image/png"
                                
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{encoded_image}",
                                    }
                                })
                        
                        content.append({
                            "type": "text",
                            "text": prompt
                        })
                        
                        messages = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": content}
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096
                )
                
                return response.choices[0].message.content
            except ImportError:
                # Fallback to using requests
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "chat2svg.github.io", 
                    "X-Title": "Chat2SVG"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.predefined_prompts["system"]},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Images are not supported in this fallback mode
                if images:
                    print("Warning: Images not supported in fallback mode for OpenRouter")
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(5, 120)
                )
                
                if response.status_code != 200:
                    print(f"Error from OpenRouter: {response.text}")
                    raise Exception(f"OpenRouter API error: {response.status_code}")
                    
                return response.json()["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error sending to OpenRouter: {str(e)}")
            raise

    def _create_payload(self, prompt: str, images: list[str] = []):
        """Creates the payload for the API request."""
        messages = {
            "role": "user",
            "content": [],
        }

        for image in images:
            base64_image = encode_image(image)
            image_content = {
                ## ChatGPT
                # "type": "image_url",
                # "image_url": {
                #     "url": base64_image,
                #     "detail": "auto",
                # }
                # Claude
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': base64_image
                }
            }
            messages["content"].append(image_content)

        messages["content"].append({
            "type": "text",
            "text": prompt,
        })

        self.past_messages.append(messages)

        return {
            "model": self.model,
            "system": self.predefined_prompts["system"],
            "messages": self.past_messages,
        }


def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # return f"data:{mime_type};base64,{encoded_string}"  # ChatGPT
        return encoded_string  # Claude
