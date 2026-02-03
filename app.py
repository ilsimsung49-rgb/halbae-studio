import streamlit as st
from google import genai
from google.genai import types
import requests
import time
import json
import os
import re
import random
import base64
from io import BytesIO
from PIL import Image

# ==========================================
# [0] CORE: ENGINE & PERSISTENCE
# ==========================================
CONFIG_FILE = "studio_config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(key, value):
    config = load_config()
    config[key] = value
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def get_api_key(service="GOOGLE_API_KEY"):
    # Try Streamlit Secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and service in st.secrets:
            return st.secrets[service]
    except:
        pass
    
    # Fallback to local config file
    config = load_config()
    return config.get(service, "")

def get_setting(key, default):
    config = load_config()
    return config.get(key, default)

# ==========================================
# [1] FORGE / A1111 BRIDGE
# ==========================================
class ForgeBridge:
    def __init__(self, base_url="http://127.0.0.1:7860"):
        self.base_url = base_url
        
    def is_active(self):
        try:
            requests.get(f"{self.base_url}/sdapi/v1/options", timeout=2)
            return True, "Connected (Forge/A1111)"
        except:
            return False, "Not Connected (Is --api enabled?)"

    def generate_image(self, prompt, neg_prompt="", width=1280, height=720):
        payload = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "steps": 25,
            "cfg_scale": 7,
            "width": width,
            "height": height,
            "sampler_name": "Euler a",
            "batch_size": 1
        }
        try:
            res = requests.post(f"{self.base_url}/sdapi/v1/txt2img", json=payload, timeout=120)
            if res.status_code == 200:
                r = res.json()
                if 'images' in r:
                    img_data = base64.b64decode(r['images'][0])
                    return img_data, None
            return None, f"Forge Error: {res.status_code} - {res.text[:100]}"
        except Exception as e:
            return None, str(e)

# ==========================================
# [2] GOOGLE ENGINE
# ==========================================

class MasterEngine:
    def __init__(self, google_key):
        self.google_key = google_key
        self.client = None
        self.forge = ForgeBridge(get_setting("FORGE_URL", "http://127.0.0.1:7860"))
        if google_key:
            self.client = genai.Client(api_key=google_key)

    def check_video_status(self, op_name):
        if not self.client: return None, "No Client"
        try:
            # Poll the operation using internal method which accepts string
            op = self.client.operations._get_videos_operation(operation_name=op_name)
            if op.done:
                if op.error: return None, f"Error: {op.error.message}"
                return op, "DONE"
            return None, "PROCESSING"
        except Exception as e:
            return None, str(e)

    def verify_connections(self):
        status = {"google_text": False, "google_image": False, "forge": False, "models": []}
        if self.client:
            try:
                # 1. LIST MODELS
                models = list(self.client.models.list())
                status['models'] = [m.name for m in models]
                
                # Check Text
                if any("gemini" in m.name for m in models): status['google_text'] = True
                
                # Check Image
                imagen_found = any("imagen" in m.name for m in models)
                if imagen_found:
                    status['google_image'] = True
                else:
                    status['model_err'] = "No 'Imagen' model found. (Check Vertex AI API)"
                    
            except Exception as e:
                status['model_err'] = str(e)

        ok, msg = self.forge.is_active()
        status['forge'] = ok
        status['forge_msg'] = msg
        return status

    def analyze_script(self, story):
        if self.client:
            try:
                # 0. Global Character Traits
                global_traits = get_setting("GLOBAL_APPEARANCE", "")
                
                # EXPERT PROMPTER PROMPT (STABLE DIFFUSION OPTIMIZED)
                prompt = f"""
                You are an expert Visual Prompt Engineer for Stable Diffusion and Midjourney.
                Your goal is to convert the story into highly detailed, KEYWORD-RICH visual prompts that yield "Masterpiece" quality images.
                
                *** MANDATORY GLOBAL TRAITS (Apply to MAIN CHARACTER): {global_traits} ***
                
                For each shot, output a JSON object with these keys:
                - "scene_num": Integer (1, 2, 3...)
                - "p_subject": HIGHLY DETAILED character description (MUST INCLUDE GLOBAL TRAITS). (e.g., "{global_traits}, Beautiful warrior woman...").
                - "p_action": Dynamic action and pose keywords. (e.g., "swinging sword, dynamic pose, shouting, muscle tension, motion blur, dramatic composition").
                - "p_background": Immersive environment details. (e.g., "ruined castle throne room, god rays, dust particles, volumetric fog, cold moonlight, cinematic lighting, 8k resolution").
                - "p_style": Camera and Art Style tags. (e.g., "Cinematic, shot on IMAX, 35mm lens, f/1.8, bokeh, hyperrealistic, Unreal Engine 5 render, dark fantasy style").
                
                CRITICAL: Do not use complete sentences using "is/are". Use COMMA-SEPARATED ADJECTIVES and NOUNS. 
                Focus on VISUALS: Texture, Lighting, Color, Lens.
                
                Story: {story}
                """
                
                model_id = get_setting("MODEL_TEXT", "gemini-1.5-flash") # Use Configured Model
                resp = self.client.models.generate_content(model=model_id, contents=prompt)
                parsed = self._parse(resp.text)
                if parsed: return parsed
                
            except Exception as e:
                print(f"Analysis Failed: {e}")
        
        return self._smart_fallback(story)

    def generate_image(self, prompt, neg, ratio="16:9"):
        mode = get_setting("GEN_MODE", "GOOGLE") 
        w, h = 1280, 720
        if ratio == "9:16": w, h = 720, 1280
        elif ratio == "1:1": w, h = 1024, 1024
        elif ratio == "2.35:1": w, h = 1536, 640
        
        if mode == "FORGE":
            if not self.forge.is_active()[0]: return None, "Forge Disconnected"
            return self.forge.generate_image(prompt, neg, w, h)
            
        if not self.client: return None, "No API Key"
        target = get_setting("MODEL_IMAGE", "imagen-3.0-generate-001")
        try:
            res = self.client.models.generate_images(
                model=target, prompt=prompt, config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio=ratio)
            )
            # Fix: Extract raw bytes from Pydantic object
            raw_img = res.generated_images[0].image
            image_bytes = raw_img.image_bytes
            return Image.open(BytesIO(image_bytes)), None
        except Exception as e:
            err_str = str(e)
            if "400" in err_str and "billed users" in err_str:
                return None, "⚠️ Google Billing Required. Please switch to 'FORGE (Local)' mode in sidebar."
            return None, f"Cloud Error: {e}"

    def generate_video(self, prompt, ratio="16:9", image_input=None):
        if not self.client: return None, "No API Key"
        target_v = get_setting("MODEL_VIDEO", "veo-3.1-generate-preview")
        
        try:
            # Prepare arguments
            kwargs = {
                "model": target_v,
                "prompt": prompt,
                "config": types.GenerateVideosConfig(aspect_ratio=ratio)
            }
            
            if image_input:
                # Convert PIL to types.Image
                buf = BytesIO()
                image_input.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                kwargs["image"] = types.Image(image_bytes=img_bytes, mime_type="image/png")
                
            res = self.client.models.generate_videos(**kwargs)
            return res, None
            
        except Exception as e:
            return None, str(e)

    def _smart_fallback(self, story):
        sentences = [s.strip() for s in re.split(r'[.\n]+', story) if len(s) > 5]
        scenes = []
        for i, sent in enumerate(sentences[:5]):
             scenes.append({
                "scene_num": i+1, "p_subject": "Character", "p_action": sent, "p_background": "Context", "p_style": "Candid"
             })
        return scenes
        
    def _parse(self, text):
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match: return json.loads(match.group())
        except: pass
        return None

# ==========================================
# [3] UI START
# ==========================================
st.set_page_config(layout="wide", page_title="HMS: PRO", page_icon="🎬")

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #eee; font-family: 'Noto Sans KR', sans-serif; }
    
    /* Sidebar visibility fix */
    [data-testid="stSidebar"] { 
        background-color: #1a1a1a; 
        border-right: 1px solid #333; 
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    .input-label { color: #00ADB5; font-size: 0.85rem; font-weight: bold; margin-bottom: 3px; }
    .shot-card { background: #1a1a1a; padding: 15px; border-left: 3px solid #00ADB5; margin-bottom: 10px; }
    .shot-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 5px; }
    .shot-tag { background: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; color: #ccc; display: inline-block; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

if "scenes" not in st.session_state: st.session_state.scenes = []
if "final_img_prompts" not in st.session_state: st.session_state.final_img_prompts = {}
if "generated_images" not in st.session_state: st.session_state.generated_images = {}

with st.sidebar:
    st.header("🎬 HALBAE STUDIO")
    step = st.radio("WORKFLOW", ["0. Home", "1. Story", "2. Analysis", "3. Image", "4. Video", "5. Final Cut", "6. Audio", "7. Settings"])
    st.divider()
    
    st.subheader("🛠️ Rendering Engine")
    curr_mode = get_setting("GEN_MODE", "FORGE")
    mode = st.radio("Source", ["GOOGLE (Cloud)", "FORGE (Local)"], index=1 if "FORGE" in curr_mode else 0)
    
    new_mode = "GOOGLE" if "GOOGLE" in mode else "FORGE"
    if new_mode != get_setting("GEN_MODE", ""):
        save_config("GEN_MODE", new_mode)
        st.rerun()
    
    if new_mode == "GOOGLE":
        api_key = get_api_key()
        if api_key and len(api_key) > 10:
            st.success("✅ API Key Connected")
        else:
            st.warning("⚠️ API Key Required")
            st.info("Set GOOGLE_API_KEY in Streamlit Cloud Secrets")
            st.caption("Settings → Secrets → Add: GOOGLE_API_KEY = 'your_key'")
    else:
        st.caption("Ensure WebUI Forge is on port 7860")
        if st.button("Check Connection"):
            engine = MasterEngine(get_api_key())
            ok, msg = engine.forge.is_active()
            if ok: st.success("Connected!")
            else: st.error("Not Found")

    with st.expander("👤 Character Studio (Global)", expanded=True):
        st.caption("Apply these traits to ALL scenes.")
        
        # 1. Global Prompt Injection
        curr_app = get_setting("GLOBAL_APPEARANCE", "")
        new_app = st.text_input("Fixed Appearance", value=curr_app, placeholder="e.g. Korean, 60s male, Hanbok, Short Hair")
        if new_app != curr_app: save_config("GLOBAL_APPEARANCE", new_app)
        
        # 2. Reference Image
        uploaded_ref = st.file_uploader("Ref Image (Visual Guide)", type=['png', 'jpg'])
        if uploaded_ref:
            st.image(uploaded_ref, caption="Character Reference")
            
    st.divider()
    with st.expander("🚀 External AI Tools"):
        st.markdown("**General**")
        st.link_button("🧠 Google AI Studio", "https://aistudio.google.com/")
        st.link_button("🤖 Grok (xAI)", "https://grok.x.ai/")
        
    st.divider()
    with st.expander("🌐 Quick Translator"):
        ko_txt = st.text_area("Korean Input", height=70, placeholder="번역할 내용을 입력하세요...")
        if st.button("Translate to Prompt"):
            if not get_api_key(): st.error("Need Google Key")
            else:
                eng = MasterEngine(get_api_key()) # Temp instance
                try:
                    target_model = get_setting("MODEL_TEXT", "gemini-1.5-flash")
                    res = eng.client.models.generate_content(
                        model=target_model, 
                        contents=f"Translate this Korean text into detailed English AI image prompts (comma separated tags). Input: {ko_txt}"
                    )
                    st.code(res.text, language="text")
                except Exception as e:
                    st.error(f"Error: {e}")
            
engine = MasterEngine(get_api_key())

# ==========================================
# 0. HOME
# ==========================================
if "0." in step:
    # Custom Title Styling with Premium Background
    st.markdown("""
    <style>
    /* Premium gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Sidebar styling for visibility */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    
    .studio-title {
        font-size: 4em;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0px;
        line-height: 1.2;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    .studio-sub {
        font-size: 1.5em;
        color: #a8b2d1;
        margin-bottom: 30px;
        font-weight: 300;
    }
    </style>
    <div class='studio-title'>HALBAE</div>
    <div class='studio-title'>MASTER STUDIO</div>
    <div class='studio-sub'>AI Cinematic Production Workflow</div>
    """, unsafe_allow_html=True)
    
    col_text, col_img = st.columns([1, 2])
    
    with col_text:
        st.markdown("""
        <div style='color: white; font-size: 1.1em;'>
        
        ### 🎬 Production Workflow
        
        **1. Analysis (Director)**
        *   Deep Script Analysis
        *   Shot List & Prompt Engineering
        
        **2. Image (Cinematographer)**
        *   Visual Development
        *   Character Consistency Control
        
        **3. Video (Camera)**
        *   Image-to-Video Animation
        *   Veo / LTX-2 Integration
        
        **4. Final Cut (Editor)**
        *   Clip Merging & Mastering
        
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start New Project", type="primary"):
            st.info("Select '1. Story' from the sidebar to begin.")

    with col_img:
        # Load home image from GitHub
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            img_url = "https://raw.githubusercontent.com/ilsimsung49-rgb/halbae-studio/main/home_bg.jpg"
            response = requests.get(img_url)
            home_img = Image.open(BytesIO(response.content))
            st.image(home_img, use_container_width=True)
        except Exception as e:
            st.info("🎬 Welcome to Halbae Master Studio")

# ==========================================
# 1. STORY (MASTER SCRIPT)
# ==========================================
elif "1." in step:
    st.subheader("STEP 1: My Master Story")
    st.caption("The original autobiography source text. Edit here, then Analyze.")
    
    story_path = "master_story.txt"
    if not os.path.exists(story_path):
        with open(story_path, "w", encoding="utf-8") as f: f.write("")
        
    with open(story_path, "r", encoding="utf-8") as f:
        current_story = f.read()
        
    new_story = st.text_area("Master Script Editor", value=current_story, height=600)
    
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("💾 Save Changes"):
            with open(story_path, "w", encoding="utf-8") as f:
                f.write(new_story)
            st.success("Story Saved!")
            time.sleep(0.5)
            st.rerun()

# ==========================================
# 2. ANALYSIS
# ==========================================
elif "2." in step:
    st.subheader("STEP 1: Professional Story Analysis")
    st.caption("AI Director will break your story into a detailed Shot List.")
    
    col_input, col_view = st.columns([1, 1.5])
    
    with col_input:
        story = st.text_area("Script / Story", value="", height=400, placeholder="Paste selected part of your story here...")
        if st.button("🎬 Run Director's Analysis", type="primary", use_container_width=True):
            with st.spinner("Director is breaking down the script..."):
                st.session_state.scenes = engine.analyze_script(story)
                st.rerun()
                
    with col_view:
        if st.session_state.scenes:
            st.success(f"Generated {len(st.session_state.scenes)} Shots")
            for s in st.session_state.scenes:
                with st.expander(f"🎬 S#{s['scene_num']}: {s.get('p_action', '')[:30]}...", expanded=True):
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"**👤 Subject**\n\n{s.get('p_subject', '-')}")
                    with c_b:
                        st.markdown(f"**🏃 Action**\n\n{s.get('p_action', '-')}")
                    st.divider()
                    c_c, c_d = st.columns(2)
                    with c_c:
                         st.markdown(f"**🏙️ Background**\n\n{s.get('p_background', '-')}")
                    with c_d:
                         st.markdown(f"**🎥 Shot Type**\n\n{s.get('p_style', 'Cinematic')}")

        else:
            st.info("👈 Enter story and click Analyze to see the breakdown.")

# ==========================================
# 3. IMAGE
# ==========================================
elif "3." in step:
    mode_label = get_setting('GEN_MODE', 'FORGE')
    st.subheader(f"STEP 2: Image Generation ({mode_label})")
    
    if not st.session_state.scenes: st.error("Analyze First")
    else:
        tabs = st.tabs([f"S{s['scene_num']}" for s in st.session_state.scenes])
        for idx, t in enumerate(tabs):
            s = st.session_state.scenes[idx]
            sn = s['scene_num']
            with t:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div class='input-label'>1. Subject</div>", unsafe_allow_html=True)
                    p1 = st.text_input(f"s{sn}_p1", value=s.get('p_subject', ''))
                    st.markdown("<div class='input-label'>4. Style</div>", unsafe_allow_html=True)
                    p4 = st.selectbox(f"s{sn}_p4", ["Cinematic", "3D Render", "Oil Painting"], index=0)
                    st.markdown("<div class='input-label'>📐 7. Ratio</div>", unsafe_allow_html=True)
                    ratio = st.selectbox(f"s{sn}_ratio", ["16:9", "9:16", "1:1", "2.35:1"])
                with c2:
                    st.markdown("<div class='input-label'>2. Action</div>", unsafe_allow_html=True)
                    p2 = st.text_input(f"s{sn}_p2", value=s.get('p_action', ''))
                    st.markdown("<div class='input-label'>5. Light/Cam</div>", unsafe_allow_html=True)
                    def_style = s.get('p_style', "Cinematic lighting")
                    p5 = st.text_input(f"s{sn}_p5", value=def_style)
                with c3:
                    st.markdown("<div class='input-label'>3. Background</div>", unsafe_allow_html=True)
                    p3 = st.text_input(f"s{sn}_p3", value=s.get('p_background', ''))
                    st.markdown("<div class='input-label'>6. Quality</div>", unsafe_allow_html=True)
                    p6 = st.text_input(f"s{sn}_p6", value="8k, masterpiece, award winning")
                
                neg = st.text_input(f"s{sn}_neg", value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry")
                final_p = f"{p1}, {p2}, {p3}, {p4} style, {p5}, {p6}"
                st.markdown("### 📋 Prompt for Forge")
                c_copy1, c_copy2 = st.columns(2)
                with c_copy1:
                    st.caption("Positive Prompt")
                    st.code(final_p, language="text")
                with c_copy2:
                    st.caption("Negative Prompt")
                    st.code(neg, language="text")
                
                if st.button(f"🎨 Generate (Scene {sn})"):
                    with st.spinner("Rendering..."):
                        res, err = engine.generate_image(final_p, neg, ratio)
                        if res:
                            st.image(res, caption=f"{mode_label} Result")
                            st.session_state.final_img_prompts[sn] = final_p
                            # SAVE IMAGE FOR VIDEO STEP
                            st.session_state.generated_images[sn] = res
                            
                            # Add Download Button
                            buf = BytesIO()
                            res.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            st.download_button(
                                label="💾 Save Image",
                                data=byte_im,
                                file_name=f"scene_{sn}_{int(time.time())}.png",
                                mime="image/png"
                            )
                        else:
                            st.error(f"Failed: {err}")
                            
        st.divider()
        st.markdown("### 🧩 External Image Tools")
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.link_button("🎨 Launch Whisk (Remix)", "https://labs.google/whisk", use_container_width=True)
        with c_t2:
            st.link_button("🧪 Launch ImageFX", "https://aitestkitchen.withgoogle.com/tools/image-fx", use_container_width=True)



# ==========================================
# 4. VIDEO
# ==========================================
elif "4." in step:
    st.subheader("STEP 3: Video Generation")
    
    # Quick Video Model Check/Display
    curr_v_model = get_setting("MODEL_VIDEO", "veo-3.1-generate-preview")
    st.caption(f"Using Video Model: `{curr_v_model}`")

    if not st.session_state.scenes: st.error("Analyze First (Step 1)")
    else:
        # Use SCENES as the source of truth, not generated images
        tabs = st.tabs([f"S{s['scene_num']}" for s in st.session_state.scenes])
        for idx, t in enumerate(tabs):
             s = st.session_state.scenes[idx]
             sn = s['scene_num']
             
             # Fallback to Analysis Data if no Image Gen happened
             base_p = st.session_state.final_img_prompts.get(sn, f"{s.get('p_subject')}, {s.get('p_background')}, {s.get('p_style')}")
             
             with t:
                 st.info("💡 Tip: Copy the final prompt below to LTX-2 / Kling / Hailuo")
                 c1, c2, c3 = st.columns(3)
                 with c1:
                     st.markdown("<div class='input-label'>1. Motion</div>", unsafe_allow_html=True)
                     v1 = st.text_input(f"v{sn}_1", placeholder="Action", value=s.get('p_action', ''))
                     st.markdown("<div class='input-label'>📐 4. Ratio</div>", unsafe_allow_html=True)
                     vr = st.selectbox(f"v{sn}_r", ["16:9", "9:16"])
                 with c2:
                     st.markdown("<div class='input-label'>2. Camera</div>", unsafe_allow_html=True)
                     v2 = st.selectbox(f"v{sn}_2", ["Static", "Zoom In", "Pan Left", "Tilt Up", "Tracking Shot"], index=1)
                 with c3:
                     st.markdown("<div class='input-label'>3. Environment</div>", unsafe_allow_html=True)
                     v3 = st.text_input(f"v{sn}_3", placeholder="Wind/Rain", value="Atmospheric lighting")
                 
                 final_v = f"{base_p}. Motion: {v1}. Camera: {v2}. Env: {v3}."
                 
                 st.markdown("### 📋 Final Video Prompt")
                 st.code(final_v, language="text")
                 
                 # CHECK FOR IMAGE
                 input_img = st.session_state.generated_images.get(sn, None)
                 if input_img:
                     st.markdown("**🖼️ Input Image (Img2Vid)**")
                     st.image(input_img, width=200)
                 
                 # Result Key
                 op_key = f"video_op_{sn}"
                 res_key = f"video_res_{sn}"
                 
                 # 1. Existing Result? Show it
                 if res_key in st.session_state:
                     uri = st.session_state[res_key]
                     st.video(uri)
                     st.markdown(f"**[Download Video]({uri})**")
                     if st.button(f"🗑️ Clear (Scene {sn})"):
                         del st.session_state[res_key]
                         st.rerun()
                 
                 # 2. Generate Button
                 if st.button(f"🎥 Generate Video (Scene {sn})"):
                     # Clear old
                     if op_key in st.session_state: del st.session_state[op_key]
                     if res_key in st.session_state: del st.session_state[res_key]
                     
                     with st.spinner(f"🚀 Initializing {curr_v_model}..."):
                         res, err = engine.generate_video(final_v, vr, image_input=input_img)
                         
                     if res:
                         op_name = res.name
                         st.session_state[op_key] = op_name
                         
                         # AUTO POLL IMMEDIATELY
                         progress_text = "🎬 Rendering in Cloud... (Please wait, high quality takes time...). "
                         my_bar = st.progress(0, text=progress_text)
                         
                         for i in range(100): # 100 * 3s = 300s (5 mins) max
                             time.sleep(3)
                             op_res, status = engine.check_video_status(op_name)
                             my_bar.progress(int((i / 100) * 100), text=f"{progress_text} ({i*3}s / 300s)")
                             
                             if status == "DONE":
                                 my_bar.progress(100, text="✨ DONE!")
                                 try:
                                     if hasattr(op_res, 'result') and op_res.result:
                                         video_uri = op_res.result.video.uri
                                         st.session_state[res_key] = video_uri
                                         st.success("Video Generated Successfully!")
                                         st.rerun() # Rerun to hit the "Existing Result" block
                                     else:
                                         st.json(op_res)
                                 except Exception as e:
                                     st.error(f"Error parsing result: {e}")

                                 break
                             elif status != "PROCESSING":
                                 # Error case
                                 if "RESOURCE_EXHAUSTED" in str(status) or "429" in str(status):
                                     st.warning("⚠️ Quota Limit Reached (429). You have used up your available Veo credits for now. Please wait or check your Google AI Studio quota.")
                                     my_bar.empty()
                                     break
                                 st.error(f"Generation Failed: {status}")
                                 my_bar.empty()
                                 break
                         else:
                             st.warning("⏱️ Still rendering... execution continues in background. Please click 'Check Status' below in a few minutes.")
                     else:
                         st.error(err if "400" not in str(err) else "Billing Required for Veo.")

                 # 3. Fallback / Manual Check (if timeout occurred or page reloaded mid-process)
                 if op_key in st.session_state and res_key not in st.session_state:
                     op_name = st.session_state[op_key]
                     st.info(f"⏳ Background Job Running: {op_name.split('/')[-1]}")
                     if st.button(f"🔄 Check Status (Poll 30s) (Scene {sn})"):
                         with st.spinner("Checking status..."):
                             for _ in range(10): # Try 10 times (30s)
                                 op_res, status = engine.check_video_status(op_name)
                                 if status == "DONE":
                                     if hasattr(op_res, 'result') and op_res.result:
                                         st.session_state[res_key] = op_res.result.video.uri
                                         st.success("Video Finished!")
                                         st.rerun()
                                 elif status != "PROCESSING":
                                      if "RESOURCE_EXHAUSTED" in str(status) or "429" in str(status):
                                          st.warning("⚠️ Quota Limit Reached (429). Please wait for quota reset.")
                                      else:
                                          st.error(f"Error: {status}")
                                      break
                                 time.sleep(3)
                             else:
                                 st.warning("Still processing... Click again soon.")

        st.divider()
        st.markdown("### 🧩 External Video Tools")
        c_v1, c_v2, c_v3 = st.columns(3)
        with c_v1:
            st.link_button("🌊 Launch Flow (Film)", "https://labs.google/flow", use_container_width=True)
        with c_v2:
            st.link_button("📹 Launch VideoFX", "https://aitestkitchen.withgoogle.com/tools/video-fx", use_container_width=True)
        with c_v3:
            st.link_button("🤖 Launch Grok (Video)", "https://grok.x.ai/", use_container_width=True)
            
        with st.expander("🤖 Grok Prompt Helper (Copy this)"):
            st.info("👇 Click the Copy button (top-right of the box) then Launch Grok")
            st.code(final_v, language="text")
            st.text_area("Or Edit Here", value=final_v, height=70, key=f"grok_p_{sn}")

# ==========================================
# 5. FINAL CUT (EDITOR)
# ==========================================
elif "5." in step:
    st.subheader("STEP 4: Final Cut (Video Editor)")
    st.info("✂️ Merge your generated clips into a final movie.")

    st.link_button("✂️ Open My CapCut Profile/Workspace", "https://www.capcut.com/profile/XtiDEV2zs5Dk0o_08Cp5NEpnUPByzHue7yXIFRr1Dyo", use_container_width=True)
    
    st.divider()
    
    col_vid, col_aud = st.columns(2)
    with col_vid:
        uploaded_clips = st.file_uploader("1. Upload Video Clips (.mp4)", type=["mp4"], accept_multiple_files=True)
    
    with col_aud:
        st.markdown("**2. Add Background Music (Suno)**")
        audio_dir = "assets/audio"
        if not os.path.exists(audio_dir): os.makedirs(audio_dir)
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))]
        
        bg_music = None
        music_source = st.radio("Music Source", ["None", "Library (Suno)", "Upload New"], horizontal=True)
        
        if music_source == "Library (Suno)":
            if audio_files:
                sel_audio = st.selectbox("Select Track", audio_files)
                bg_music = os.path.join(audio_dir, sel_audio)
                st.audio(bg_music)
            else:
                st.warning("No files in Library. Go to Step 6 to upload.")
        elif music_source == "Upload New":
            up_audio = st.file_uploader("Upload Audio", type=['mp3', 'wav'])
            if up_audio:
                # Save temp
                with open(os.path.join(audio_dir, up_audio.name), "wb") as f:
                    f.write(up_audio.getbuffer())
                bg_music = os.path.join(audio_dir, up_audio.name)
                st.success("Loaded!")

    if uploaded_clips:
        st.write(f"🎥 Clips Ready: {len(uploaded_clips)}")
        if st.button("🎬 Merge Clips"):
            with st.spinner("Merging..."):
                try:
                    import moviepy as mp
                    import tempfile
                    
                    clips = []
                    temp_files = []
                    
                    for uf in uploaded_clips:
                        # Save to temp
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(uf.read())
                        tfile.close()
                        temp_files.append(tfile.name)
                        clips.append(mp.VideoFileClip(tfile.name))
                    
                    if not clips:
                        st.error("No clips loaded.")
                    else:
                        final_clip = mp.concatenate_videoclips(clips, method="compose")
                        
                        # Add Audio if selected
                        if bg_music and os.path.exists(bg_music):
                            st.info(f"🎵 Adding Music: {os.path.basename(bg_music)}")
                            audio_clip = mp.AudioFileClip(bg_music)
                            # Loop or Trim? Let's Trim/Loop to fit
                            if audio_clip.duration < final_clip.duration:
                                # Simple option: loop it? Or just let it end. Let's loop.
                                # audio_clip = audio_clip.fx(mp.vfx.loop, duration=final_clip.duration) # v2 syntax might vary
                                pass # Keep simple for now, just set it
                            
                            # Trim to video length
                            audio_clip = audio_clip.subclipped(0, min(audio_clip.duration, final_clip.duration))
                            final_clip = final_clip.with_audio(audio_clip) # v2 syntax: .with_audio instead of .set_audio? check docs or try .with_audio

                        out_path = f"final_movie_{int(time.time())}.mp4"
                        final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
                        
                        # Cleanup
                        for c in clips: c.close()
                        # for t in temp_files: os.unlink(t) # Keep temp for safety or delete
                        
                        st.success("✨ Merge Complete!")
                        st.video(out_path)
                        
                        with open(out_path, "rb") as f:
                            st.download_button("💾 Download Final Movie", f, file_name="final_movie.mp4")
                        
                except Exception as e:
                    st.error(f"Merge Failed: {e}")

# ==========================================
# 6. AUDIO (Placeholder)
# ==========================================
elif "6." in step:
    st.subheader("STEP 5: Audio Studio")
    
    # 1. External Tools
    st.markdown("### 🎹 AI Audio Tools")
    st.markdown("### 🎹 AI Audio Tools")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.link_button("🎵 Launch MusicFX", "https://aitestkitchen.withgoogle.com/tools/music-fx", use_container_width=True)
    with c_m2:
        st.link_button("🎧 Go to My Suno Profile", "https://suno.com/@jason889", use_container_width=True)
    
    st.divider()
    
    # 2. Suno Upload & Manager
    st.markdown("### 🎧 Suno Music Manager")
    st.caption("Upload your Suno creations here for safe keeping.")
    
    # Ensure directory exists
    audio_dir = "assets/audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        
    # Uploader
    uploaded_files = st.file_uploader("Upload Suno Music (.mp3, .wav)", type=['mp3', 'wav'], accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            file_path = os.path.join(audio_dir, uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} tracks!")
        time.sleep(1)
        st.rerun()

    # Playlist
    files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))]
    if files:
        st.markdown(f"**📚 Library ({len(files)} tracks)**")
        for f in files:
            col_play, col_del = st.columns([4, 1])
            with col_play:
                st.audio(os.path.join(audio_dir, f), format='audio/mp3')
                st.caption(f"track: {f}")
            with col_del:
                if st.button("🗑️", key=f"del_{f}"):
                    os.remove(os.path.join(audio_dir, f))
                    st.rerun()
    else:
        st.info("No music uploaded yet.")

# ==========================================
# 7. SETTINGS
# ==========================================
elif "7." in step:
    st.subheader("⚙️ Engine Status & Settings")
    
    st.markdown("### 🔍 Model Diagnostic")
    
    # Auto-load status if enabled before
    if "conn_status" not in st.session_state:
        st.session_state.conn_status = None

    if st.button("Check Available Models"):
        st.session_state.conn_status = engine.verify_connections()
        
    if st.session_state.conn_status:
        s = st.session_state.conn_status
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Google API**: {'✅ Active' if s['google_text'] else '❌ Inactive'}")
            st.markdown(f"**WebUI Forge**: {'✅ ' + s['forge_msg'] if s['forge'] else '❌ ' + s['forge_msg']}")
            
        st.divider()
        st.subheader("1. 🖼️ Cloud Image Model (Imagen)")
        if 'models' in s and s['models']:
            imagen_models = [m for m in s['models'] if "imagen" in m]
            if imagen_models:
                current_model = get_setting("MODEL_IMAGE", "imagen-3.0-generate-001")
                if current_model not in imagen_models:
                    current_model = imagen_models[0]
                    save_config("MODEL_IMAGE", current_model)
                
                sel_img = st.selectbox("Select Image Model", imagen_models, index=imagen_models.index(current_model) if current_model in imagen_models else 0, key="set_img")
                if sel_img != current_model:
                    save_config("MODEL_IMAGE", sel_img)
                    st.success(f"Saved: {sel_img}")
                    time.sleep(0.1) 
                    st.rerun()
            else:
                st.error("No 'Imagen' models found.")

            st.divider()
            st.subheader("2. 📝 Cloud Text Model (Gemini)")
            gemini_models = [m for m in s['models'] if "gemini" in m and "vision" not in m]
            if gemini_models:
                current_text = get_setting("MODEL_TEXT", "gemini-1.5-flash")
                if current_text not in gemini_models:
                    flash_models = [m for m in gemini_models if 'flash' in m]
                    current_text = flash_models[0] if flash_models else gemini_models[0]
                    save_config("MODEL_TEXT", current_text)
                
                sel_text = st.selectbox("Select Analysis Model", gemini_models, index=gemini_models.index(current_text) if current_text in gemini_models else 0, key="set_txt")
                if sel_text != current_text:
                    save_config("MODEL_TEXT", sel_text)
                    st.success(f"Saved: {sel_text}")
                    time.sleep(0.1)
                    st.rerun()
            else:
                st.warning("No standard Gemini text models found.")

            st.divider()
            st.subheader("3. 🎥 Cloud Video Model (Veo)")
            veo_models = [m for m in s['models'] if "veo" in m]
            if veo_models:
                current_video = get_setting("MODEL_VIDEO", "veo-3.1-generate-preview")
                if current_video not in veo_models:
                    # Prioritize 3.1 > 3.0 > 2.0
                    sorted_veo = sorted(veo_models, key=lambda x: "3.1" in x, reverse=True)
                    current_video = sorted_veo[0]
                    save_config("MODEL_VIDEO", current_video)
                
                sel_video = st.selectbox("Select Video Model", veo_models, index=veo_models.index(current_video) if current_video in veo_models else 0, key="set_vid")
                if sel_video != current_video:
                    save_config("MODEL_VIDEO", sel_video)
                    st.success(f"Saved: {sel_video}")
                    time.sleep(0.1)
                    st.rerun()
            else:
                st.warning("No 'Veo' video models found (Video Gen will fail without Tier 1 access).")

        else:
            st.error(f"Failed to list models: {s.get('model_err', 'Unknown Error')}")
