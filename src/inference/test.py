import os
import glob
import numpy as np
import lovely_tensors as lt
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import base64
from datetime import datetime
from io import BytesIO
from torchvision import transforms

from src.models.pixio.dpt import DPTDepth
from src.utils import disp_to_depth
from src.config.conf import Conf

def image_to_base64(image):
    """Convert PIL Image to base64 string for HTML embedding"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def numpy_to_base64(array):
    """Convert numpy array to base64 string"""
    im = pil.fromarray(array)
    return image_to_base64(im)

def create_html_report(results, output_directory, conf):
    """Create a comprehensive HTML report with all predictions"""
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JEPADepth Inference Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            
            .header .subtitle {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .header .timestamp {{
                margin-top: 15px;
                font-size: 0.9em;
                opacity: 0.8;
            }}
            
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
                border-bottom: 3px solid #667eea;
            }}
            
            .summary-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .summary-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.15);
            }}
            
            .summary-card .icon {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            
            .summary-card .value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }}
            
            .summary-card .label {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .config-section {{
                padding: 30px;
                background: #fff;
            }}
            
            .config-section h2 {{
                color: #667eea;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }}
            
            .config-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            
            .config-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            
            .config-item .key {{
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }}
            
            .config-item .value {{
                color: #666;
                font-family: 'Courier New', monospace;
            }}
            
            .results-section {{
                padding: 30px;
            }}
            
            .results-section h2 {{
                color: #667eea;
                margin-bottom: 30px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }}
            
            .result-item {{
                margin-bottom: 50px;
                background: #f8f9fa;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .result-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .result-header h3 {{
                font-size: 1.3em;
                margin: 0;
            }}
            
            .result-header .index {{
                background: rgba(255,255,255,0.2);
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }}
            
            .result-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                padding: 20px 30px;
                background: white;
            }}
            
            .stat-box {{
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            
            .stat-box .stat-label {{
                color: #666;
                font-size: 0.85em;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .stat-box .stat-value {{
                color: #667eea;
                font-size: 1.3em;
                font-weight: bold;
            }}
            
            .result-images {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                padding: 30px;
                background: white;
            }}
            
            .image-container {{
                position: relative;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            
            .image-container:hover {{
                transform: scale(1.02);
                box-shadow: 0 8px 12px rgba(0,0,0,0.2);
            }}
            
            .image-container img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            
            .image-label {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                background: linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%);
                color: white;
                padding: 15px;
                font-weight: bold;
                font-size: 1.1em;
            }}
            
            .depth-comparison {{
                padding: 30px;
                background: white;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            
            .depth-viz {{
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .depth-viz img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            
            .depth-viz .label {{
                background: #667eea;
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
            }}
            
            .colorbar-legend {{
                padding: 20px 30px;
                background: #f8f9fa;
                margin: 20px 30px;
                border-radius: 10px;
                text-align: center;
            }}
            
            .colorbar-legend h4 {{
                margin-bottom: 15px;
                color: #333;
            }}
            
            .colorbar {{
                height: 30px;
                background: linear-gradient(to right, 
                    #440154, #482777, #3e4989, #31688e, #26828e,
                    #1f9e89, #35b779, #6ece58, #b5de2b, #fde724);
                border-radius: 5px;
                margin: 10px 0;
            }}
            
            .colorbar-labels {{
                display: flex;
                justify-content: space-between;
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }}
            
            .footer {{
                background: #2c3e50;
                color: white;
                text-align: center;
                padding: 30px;
                font-size: 0.9em;
            }}
            
            .footer a {{
                color: #667eea;
                text-decoration: none;
            }}
            
            .footer a:hover {{
                text-decoration: underline;
            }}
            
            @media (max-width: 768px) {{
                .summary {{
                    grid-template-columns: 1fr;
                }}
                
                .result-images {{
                    grid-template-columns: 1fr;
                }}
                
                .depth-comparison {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .progress-bar {{
                width: 100%;
                height: 4px;
                background: #e0e0e0;
                position: relative;
                overflow: hidden;
            }}
            
            .progress-bar::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>🎯 JEPADepth Inference Report</h1>
                <div class="subtitle">Monocular Depth Estimation Results</div>
                <div class="timestamp">Generated on {timestamp}</div>
            </div>
            
            <div class="progress-bar"></div>
            
            <!-- Summary Section -->
            <div class="summary">
                <div class="summary-card">
                    <div class="icon">📷</div>
                    <div class="value">{num_images}</div>
                    <div class="label">Images Processed</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🖼️</div>
                    <div class="value">{input_resolution}</div>
                    <div class="label">Input Resolution</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🤖</div>
                    <div class="value">{model_name}</div>
                    <div class="label">Model Architecture</div>
                </div>
                <div class="summary-card">
                    <div class="icon">⚡</div>
                    <div class="value">{device}</div>
                    <div class="label">Device</div>
                </div>
            </div>
            
            <!-- Configuration Section -->
            <div class="config-section">
                <h2>⚙️ Configuration</h2>
                <div class="config-grid">
                    {config_items}
                </div>
            </div>
            
            <!-- Results Section -->
            <div class="results-section">
                <h2>📊 Prediction Results</h2>
                
                <!-- Colorbar Legend -->
                <div class="colorbar-legend">
                    <h4>Depth Colormap Legend (Viridis)</h4>
                    <div class="colorbar"></div>
                    <div class="colorbar-labels">
                        <span>Near (Low Disparity)</span>
                        <span>Far (High Disparity)</span>
                    </div>
                    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                        Warmer colors (yellow/green) represent closer objects, cooler colors (purple/blue) represent farther objects
                    </p>
                </div>
                
                {results_html}
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>Generated by <strong>JEPADepth</strong> - Self-Supervised Monocular Depth Estimation</p>
                <p style="margin-top: 10px;">
                    <a href="https://github.com/yourusername/JEPADepth" target="_blank">GitHub Repository</a> | 
                    <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">Paper</a>
                </p>
                <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.8;">
                    © 2024 JEPADepth Project. All rights reserved.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate config items HTML
    config_items_html = ""
    config_display = {
        "Model Name": conf.get('model_name', 'N/A'),
        "Input Height": conf['im_sz'][0],
        "Input Width": conf['im_sz'][1],
        "Weights Path": conf.get('weights_path', 'N/A'),
        "Device": "CUDA" if torch.cuda.is_available() else "CPU",
        "Image Extension": conf.get('image_extension_inference', 'jpg'),
        "Min Depth": "0.1m",
        "Max Depth": "100m",
    }
    
    for key, value in config_display.items():
        config_items_html += f"""
        <div class="config-item">
            <div class="key">{key}</div>
            <div class="value">{value}</div>
        </div>
        """
    
    # Generate results HTML
    results_html = ""
    for idx, result in enumerate(results):
        result_html = f"""
        <div class="result-item">
            <div class="result-header">
                <h3>📸 {result['filename']}</h3>
                <div class="index">#{idx + 1}</div>
            </div>
            
            <div class="result-stats">
                <div class="stat-box">
                    <div class="stat-label">Original Size</div>
                    <div class="stat-value">{result['original_size'][0]} × {result['original_size'][1]}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Model Input</div>
                    <div class="stat-value">{result['model_input_size'][0]} × {result['model_input_size'][1]}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Min Disparity</div>
                    <div class="stat-value">{result['disp_min']:.4f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Max Disparity</div>
                    <div class="stat-value">{result['disp_max']:.4f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Mean Disparity</div>
                    <div class="stat-value">{result['disp_mean']:.4f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">95th Percentile</div>
                    <div class="stat-value">{result['disp_95th']:.4f}</div>
                </div>
            </div>
            
            <div class="result-images">
                <div class="image-container">
                    <div class="image-label">🖼️ Original Image</div>
                    <img src="{result['original_image_b64']}" alt="Original Image">
                </div>
                <div class="image-container">
                    <div class="image-label">🎨 Depth Prediction (Colorized)</div>
                    <img src="{result['colored_depth_b64']}" alt="Colored Depth">
                </div>
            </div>
            
            <div class="depth-comparison">
                <div class="depth-viz">
                    <div class="label">Raw Disparity Map</div>
                    <img src="{result['raw_disparity_b64']}" alt="Raw Disparity">
                </div>
                <div class="depth-viz">
                    <div class="label">Scaled Disparity Map</div>
                    <img src="{result['scaled_disparity_b64']}" alt="Scaled Disparity">
                </div>
            </div>
        </div>
        """
        results_html += result_html
    
    # Fill template
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_images=len(results),
        input_resolution=f"{conf['im_sz'][1]}×{conf['im_sz'][0]}",
        model_name=conf.get('model_name', 'N/A'),
        device="CUDA" if torch.cuda.is_available() else "CPU",
        config_items=config_items_html,
        results_html=results_html
    )
    
    # Save HTML report
    report_path = os.path.join(output_directory, "inference_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML Report saved to: {report_path}")
    return report_path

def test_simple(conf):
    """
        Function to predict depth map(s) for a single image or folder of images as input.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_input_height = conf['im_sz'][0]
    model_input_width = conf['im_sz'][1]

    # Preparing model
    if conf.get('model_name').startswith("pixio"):
        model = DPTDepth(conf['pixio']['encoder'], conf['pixio']['pretrained_ckp'])
    else:
        raise NotImplementedError("Model not implemented for evaluation!")
    model.from_pretrained(weights_path=conf['weights_path'], device=device)
    model.to(device)
    model.eval()

    # Finding input image(s)
    if os.path.isfile(conf['image_path_inference']):
        # Testing on a single image
        paths = [conf['image_path_inference']]
        output_directory = os.path.join(os.path.dirname(__file__), "output_images")
    elif os.path.isdir(conf['image_path_inference']):
        # Testing on a folder of images
        paths = glob.glob(os.path.join(conf['image_path_inference'], '*.{}'.format(conf['image_extension_inference'])))
        output_directory = os.path.join(os.path.dirname(__file__), "output_images")
    else:
        raise Exception("Can not find conf['image_path_inference']: {}".format(conf['image_path_inference']))

    print("-> Predicting on {:d} test image(s)".format(len(paths)))

    # Store results for HTML report
    results = []

    # Predicting on each image
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image_resized = input_image.resize((model_input_width, model_input_height), pil.Resampling.LANCZOS)
            input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)


            # Prediction
            input_tensor = input_tensor.to(device)
            output_disparity_map = model(input_tensor)[0]
            scaled_disp, depth = disp_to_depth(output_disparity_map, conf['min_depth'], conf['max_depth'])

            output_disparity_map_resized = torch.nn.functional.interpolate(output_disparity_map, (original_height, original_width), mode="bicubic", align_corners=True)
            depth_resized = torch.nn.functional.interpolate(depth, (original_height, original_width), mode="bicubic", align_corners=True)
            scaled_disp_resized = torch.nn.functional.interpolate(scaled_disp, (original_height, original_width), mode="bicubic", align_corners=True)

            # Saving color mapped depth image
            output_disparity_map_resized_np = output_disparity_map_resized.squeeze().cpu().numpy()
            scaled_disp_resized_np = scaled_disp_resized.squeeze().cpu().numpy()
            
            vmax = np.percentile(output_disparity_map_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=output_disparity_map_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
            color_depth_map = (mapper.to_rgba(output_disparity_map_resized_np)[:, :, :3] * 255).astype(np.uint8)

            # Create visualizations for raw and scaled disparity
            normalizer_raw = mpl.colors.Normalize(vmin=output_disparity_map_resized_np.min(), vmax=output_disparity_map_resized_np.max())
            mapper_raw = cm.ScalarMappable(norm=normalizer_raw, cmap='plasma')
            raw_disp_viz = (mapper_raw.to_rgba(output_disparity_map_resized_np)[:, :, :3] * 255).astype(np.uint8)
            
            normalizer_scaled = mpl.colors.Normalize(vmin=scaled_disp_resized_np.min(), vmax=scaled_disp_resized_np.max())
            mapper_scaled = cm.ScalarMappable(norm=normalizer_scaled, cmap='plasma')
            scaled_disp_viz = (mapper_scaled.to_rgba(scaled_disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            # Store result data for HTML report
            result_data = {
                'filename': os.path.basename(image_path),
                'original_size': (original_width, original_height),
                'model_input_size': (model_input_width, model_input_height),
                'disp_min': float(output_disparity_map_resized_np.min()),
                'disp_max': float(output_disparity_map_resized_np.max()),
                'disp_mean': float(output_disparity_map_resized_np.mean()),
                'disp_95th': float(vmax),
                'original_image_b64': image_to_base64(input_image),
                'colored_depth_b64': numpy_to_base64(color_depth_map),
                'raw_disparity_b64': numpy_to_base64(raw_disp_viz),
                'scaled_disparity_b64': numpy_to_base64(scaled_disp_viz),
            }
            results.append(result_data)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(paths)))
            print("                                         {}".format(output_directory))

    # Generate HTML report
    print("\n-> Generating HTML report...")
    create_html_report(results, output_directory, conf)

    print('-> Done!')

if __name__ == '__main__':

    lt.monkey_patch()

    conf = Conf().conf
    test_simple(conf)

