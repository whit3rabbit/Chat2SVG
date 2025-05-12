from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from typing import Dict, Any, List, Optional

from models import LLMProvider, ProviderSettings, AppSettings
from llm_service import get_llm_service

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

process_pool = {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT)
env["PYTHONUNBUFFERED"] = "1"

# Load environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# Set up environment variables for LLM providers
# We keep this for backward compatibility
env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
# Add other providers' environment variables
for key in os.environ:
    if key.startswith(("OPENAI_", "ANTHROPIC_", "OPENROUTER_", "LOCAL_LLM_")):
        env[key] = os.environ[key]


async def run_stage(websocket: WebSocket, stage: int, prompt: str, provider: Optional[str] = None):
    target = prompt.replace(" ", "_")
    output_path = PROJECT_ROOT / "output"
    output_folder = "example_generation/" + target

    # Get provider settings
    llm_service = get_llm_service()
    provider_enum = LLMProvider(provider) if provider else None
    
    # If provider specified but not available, return error
    if provider_enum and provider_enum not in llm_service.settings.providers:
        await websocket.send_json({
            "stage": stage,
            "status": "error",
            "output": f"Provider {provider} not configured. Please add API key in settings."
        })
        return
        
    # If provider specified, get model from provider settings
    model = None
    if provider_enum:
        provider_settings = llm_service.settings.providers[provider_enum]
        model = provider_settings.model

    stage_scripts = {
        1: [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "1_template_generation/main.py"),
            "--target", target,
            "--output_path", str(output_path),
            "--output_folder", output_folder,
            "--system_path", str(PROJECT_ROOT),
            "--model", model or "claude-3-5-sonnet-20240620",
            "--prompt", prompt
        ],
        2: [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "2_detail_enhancement/main.py"),
            "--target", target,
            "--output_path", str(output_path),
            "--output_folder", output_folder,
            "--seed", "0",
            "--num_images_per_prompt", "4",
            "--strength", "1.0",
            "--thresh_iou", "0.4",
            "--prompt", prompt,
            "--diffusion_model_id", str(PROJECT_ROOT / "2_detail_enhancement/models/aamXLAnimeMix_v10.safetensors"),
            "--sam_checkpoint", str(PROJECT_ROOT / "2_detail_enhancement/models/sam_vit_h_4b8939.pth")
        ],
        3: [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "3_svg_optimization/main.py"),
            "--target", target,
            "--svg_folder", str(output_path) + "/" + str(output_folder),
            "--system_path", str(PROJECT_ROOT / "3_svg_optimization"),
            "--output_size", "256",
            "--smoothness_weight_img", "2.0",
            "--mse_loss_weight_img", "2000.0",
            "--kl_weight_img", "0.2",
            "--vae_optim_config", str(PROJECT_ROOT / "3_svg_optimization/configs/vae_config_cmd_10.yaml"),
            "--vae_pretrained_path", str(PROJECT_ROOT / "3_svg_optimization/vae_model/cmd_10.pth")
        ]
    }

    try:
        proc = await asyncio.create_subprocess_exec(
            *stage_scripts[stage],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT)
        )

        process_pool[stage] = proc

        while True:
            try:
                line = await proc.stdout.read(1024)
                if not line:
                    if proc.returncode is not None:
                        break
                    await asyncio.sleep(0.1)
                    continue

                decoded = line.decode('utf-8', errors='replace')
                decoded = decoded.replace('\r\n', '\n').rstrip('\n')

                for output_line in decoded.split('\n'):
                    output_line = output_line.strip()
                    if not output_line:
                        continue

                    await websocket.send_json({
                        "stage": stage,
                        "status": "running",
                        "output": output_line
                    })

                    if stage == 1 and "The best SVG is:" in output_line:
                        svg_filename = output_line.split(": ")[-1].strip()
                        await websocket.send_json({
                            "stage": 1,
                            "status": "svg_ready",
                            "svg_file": output_folder + "/stage_1/svg_logs/" + svg_filename
                        })

                    if stage == 1 and output_line == "Stage 1 Done!":
                        await websocket.send_json({
                            "stage": 1,
                            "status": "completed",
                            "output": "Stage 1 Finished"
                        })

                    if stage == 2 and "The best SVG is:" in output_line:
                        svg_filename = output_line.split(": ")[-1].strip()
                        await websocket.send_json({
                            "stage": 2,
                            "status": "svg_ready2",
                            "svg_file": svg_filename
                        })

                    if stage == 2 and output_line == "Stage 2 Done!":
                        await websocket.send_json({
                            "stage": 2,
                            "status": "completed",
                            "output": "Stage 2 Finished"
                        })

                    if stage == 3 and "The best SVG is:" in output_line:
                        svg_filename = output_line.split(": ")[-1].strip()
                        await websocket.send_json({
                            "stage": 3,
                            "status": "svg_ready3",
                            "svg_file": svg_filename
                        })

                    if stage == 3 and output_line == "Stage 3 Done!":
                        await websocket.send_json({
                            "stage": 3,
                            "status": "completed",
                            "output": "Stage 3 Finished"
                        })

            except (asyncio.CancelledError, ConnectionResetError):
                proc.terminate()
                raise
            except Exception as e:
                print(f"error: {str(e)}")
                break

        exit_code = await proc.wait()

        if exit_code != 0:
            await websocket.send_json({
                "stage": stage,
                "status": "error",
                "output": f"Stage {stage} failed with code {exit_code}"
            })
        else:
            await websocket.send_json({
                "stage": stage,
                "status": "completed"
            })

    except asyncio.CancelledError:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
        raise
    except Exception as e:
        await websocket.send_json({
            "stage": stage,
            "status": "error",
            "output": f"Stage {stage} crashed: {str(e)}"
        })


@app.websocket("/ws/generate")
async def websocket_generator(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        params = json.loads(data)
        prompt = params["prompt"]
        start_from = params.get("startFrom", 1)
        provider = params.get("provider", None)  # Optional provider parameter
        stages_to_run = [s for s in [1, 2, 3] if s >= start_from]

        for stage in stages_to_run:
            await websocket.send_json({
                "stage": stage,
                "status": "started",
                "output": f"Stage {stage} started"
            })

            try:
                await asyncio.wait_for(
                    run_stage(websocket, stage, prompt, provider),
                    timeout=2400
                )
            except asyncio.TimeoutError:
                proc = process_pool.get(stage)
                if proc and proc.returncode is None:
                    proc.terminate()
                    await proc.wait()

                await websocket.send_json({
                    "stage": stage,
                    "status": "error",
                    "output": f"Stage {stage} timed out after 40 minutes"
                })
                break

    except json.JSONDecodeError:
        await websocket.send_json({
            "error": "Invalid JSON format"
        })
    except Exception as e:
        await websocket.send_json({
            "error": str(e)
        })
    finally:
        await websocket.close()


@app.get("/api/svg/{file_path:path}")
async def get_svg(file_path: str):
    svg_path = PROJECT_ROOT / "output" / file_path

    # Security checks
    if any(c in file_path for c in {'..', '//', '\\', ':'}):
        raise HTTPException(400, "Illegal file path")

    if not svg_path.exists():
        raise HTTPException(404, "File not found")

    if svg_path.suffix.lower() != '.svg':
        raise HTTPException(400, "Not an SVG file")

    return {
        "status": "ok",
        "content": svg_path.read_text(encoding='utf-8')
    }


@app.get("/api/settings")
async def get_settings():
    """Get current LLM provider settings"""
    llm_service = get_llm_service()
    
    # Create a safe version of settings without API keys
    safe_settings = {
        "default_provider": llm_service.settings.default_provider,
        "providers": {}
    }
    
    for provider, settings in llm_service.settings.providers.items():
        safe_settings["providers"][provider] = {
            "provider": settings.provider,
            "model": settings.model,
            "api_base": settings.api_base,
            # Don't return the actual API key, just whether it's set
            "has_api_key": bool(settings.api_key)
        }
    
    return safe_settings


@app.get("/api/models")
async def get_available_models():
    """Get predefined models available for each provider"""
    # Define popular models for each provider
    models = {
        "openai": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ],
        "anthropic": [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        "openrouter": [
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet",
            "anthropic/claude-3-opus",
            "mistralai/mistral-large",
            "google/gemini-1.5-pro"
        ],
        "local": [
            "default"  # Local model names depend on setup
        ]
    }
    
    return models


@app.post("/api/settings/provider")
async def update_provider_settings(settings: Dict[str, Any]):
    """Update provider settings"""
    try:
        # Update environment with the new settings
        if "provider" not in settings:
            raise HTTPException(400, "Provider parameter is required")
            
        provider = settings["provider"]
        
        # Update environment variables based on provider
        if provider == "openai":
            prefix = "OPENAI_"
        elif provider == "anthropic":
            prefix = "ANTHROPIC_"
        elif provider == "openrouter":
            prefix = "OPENROUTER_"
        elif provider == "local":
            prefix = "LOCAL_LLM_"
        else:
            raise HTTPException(400, f"Unsupported provider: {provider}")
            
        # Update API key if provided
        if "api_key" in settings and settings["api_key"]:
            os.environ[f"{prefix}API_KEY"] = settings["api_key"]
            env[f"{prefix}API_KEY"] = settings["api_key"]
            
        # Update API base if provided
        if "api_base" in settings and settings["api_base"]:
            os.environ[f"{prefix}API_BASE"] = settings["api_base"]
            env[f"{prefix}API_BASE"] = settings["api_base"]
            
        # Update model if provided
        if "model" in settings and settings["model"]:
            os.environ[f"{prefix}MODEL"] = settings["model"]
            env[f"{prefix}MODEL"] = settings["model"]
            
        # Set as default provider if requested
        if "set_default" in settings and settings["set_default"]:
            os.environ["DEFAULT_LLM_PROVIDER"] = provider
            env["DEFAULT_LLM_PROVIDER"] = provider
            
        # Reinitialize LLM service with updated settings
        global _llm_service_instance
        from llm_service import _llm_service_instance
        _llm_service_instance = None  # Force reinitialization
        
        # Get updated settings
        llm_service = get_llm_service()
        
        return {
            "status": "ok",
            "message": f"Updated settings for {provider}",
            "default_provider": llm_service.settings.default_provider
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to update settings: {str(e)}")


@app.get("/api/test_connection/{provider}")
async def test_provider_connection(provider: str):
    """Test connection to a provider"""
    try:
        llm_service = get_llm_service()
        
        # Check if provider exists
        if LLMProvider(provider) not in llm_service.settings.providers:
            return {
                "status": "error",
                "message": f"Provider {provider} not configured"
            }
            
        # Test simple completion
        response = await llm_service.generate_completion(
            provider=LLMProvider(provider),
            prompt="Hello, please respond with just the word 'Connected' if you can see this message.",
            max_tokens=10
        )
        
        return {
            "status": "ok",
            "message": "Connection successful",
            "model": response["model"],
            "response": response["content"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Connection failed: {str(e)}"
        }
