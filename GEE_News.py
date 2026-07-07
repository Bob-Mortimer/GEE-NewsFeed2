import streamlit as st
import ee
import folium
import streamlit.components.v1 as components
from datetime import date, timedelta
import requests
from google.oauth2 import service_account

# =========================================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# =========================================================================
# Set wide layout and minimize Streamlit's default block padding via CSS
st.set_page_config(layout="wide", page_title="Geospatial Intelligence Dashboard")

st.markdown("""
    <style>
        /* Compress the top and bottom padding of the main Streamlit container */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }
        /* Remove border and padding from iframes to kill dead space */
        iframe {
            border: none !important;
            margin: 0 !important;
            padding: 0 !important;
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_ee():
    try:
        service_account_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
        ee.Initialize(credentials=scoped_credentials, project=service_account_info["project_id"])
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.stop()

initialize_ee()

# =========================================================================
# 2. CORE INTELLIGENCE LOGIC
# =========================================================================
@st.cache_data(ttl=3600)
def get_coordinates_opencage(query):
    try:
        api_key = st.secrets.get("opencage_api_key", None)
        if not api_key:
            return None, None
        
        url = f"https://api.opencagedata.com/geocode/v1/json?q={query}&key={api_key}&limit=1"
        response = requests.get(url, timeout=5).json()
        if response.get('results'):
            lat = response['results'][0]['geometry']['lat']
            lon = response['results'][0]['geometry']['lng']
            return lat, lon
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None, None

def fetch_intelligence_news(user_query):
    api_key = st.secrets.get("newsapi_key", None)
    if not api_key:
        st.error("Missing NewsAPI Key in secrets.")
        return []
        
    keywords = '(military OR cyber OR politics OR political OR president OR "prime minister" OR conflict OR flashpoint)'
    
    if user_query and user_query.strip():
        q = f"({keywords}) AND ({user_query.strip()})"
    else:
        q = keywords
        
    url = f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=relevancy&apiKey={api_key}&pageSize=10"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get('articles', [])
    except Exception as e:
        st.error(f"News fetch failed: {e}")
    return []

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)

def add_ee_layer(m, ee_image_object, vis_params, name):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=name,
            overlay=True,
            control=True
        ).add_to(m)
    except Exception:
        pass

def create_maps(lat, lon, start1, end1, start2, end2, sensitivity):
    point = ee.Geometry.Point([lon, lat])
    
    # Optical Data
    s2_base = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
               .filterBounds(point).filterDate(start1, end1)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
               .map(mask_s2_clouds).median())
    s2_comp = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
               .filterBounds(point).filterDate(start2, end2)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
               .map(mask_s2_clouds).median())
               
    b1, b2 = s2_base.select('B4'), s2_comp.select('B4')
    L, c1, c2 = 10000, (0.01 * 10000)**2, (0.03 * 10000)**2
    kernel = ee.Kernel.square(3)
    
    mu1, mu2 = b1.reduceNeighborhood(ee.Reducer.mean(), kernel), b2.reduceNeighborhood(ee.Reducer.mean(), kernel)
    var1, var2 = b1.reduceNeighborhood(ee.Reducer.variance(), kernel), b2.reduceNeighborhood(ee.Reducer.variance(), kernel)
    cov12 = b1.multiply(b2).reduceNeighborhood(ee.Reducer.mean(), kernel).subtract(mu1.multiply(mu2))
    
    num = (mu1.multiply(mu2).multiply(2).add(c1)).multiply(cov12.multiply(2).add(c2))
    den = (mu1.pow(2).add(mu2.pow(2)).add(c1)).multiply(var1.add(var2).add(c2))
    ssim = num.divide(den)
    ssim_mask = ssim.lt(1.0 - (sensitivity * 0.5))
    s2_diff_red_dots = ssim_mask.updateMask(ssim_mask)
    
    # SAR Data
    s1_base = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterBounds(point).filterDate(start1, end1)
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .select('VV').median())
    s1_comp = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterBounds(point).filterDate(start2, end2)
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .select('VV').median())

    s1_diff_abs = s1_comp.subtract(s1_base).abs()
    s1_raw_mask = s1_diff_abs.gt(4.5 - (sensitivity * 3))
    s1_clean_mask = s1_raw_mask.focal_mode(radius=30, units='meters')
    s1_red_dots = s1_clean_mask.updateMask(s1_clean_mask)

    vis_s2 = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
    vis_diff = {'min': 1, 'max': 1, 'palette': ['FF0000']} 
    vis_s1 = {'min': -25, 'max': 0}
    
    map_kwargs = {'location': [lat, lon], 'zoom_start': 13, 'tiles': 'CartoDB dark_matter', 'width': '100%', 'height': '100%'}
    
    maps = []
    for layer_setup in [
        (s2_base, vis_s2, 'S2 Baseline'),
        (s2_comp, vis_s2, 'S2 Comparison', s2_diff_red_dots, vis_diff, 'Optical Change (SSIM)'),
        (s1_base, vis_s1, 'S1 Baseline'),
        (s1_comp, vis_s1, 'S1 Comparison', s1_red_dots, vis_diff, 'SAR Change (Cleaned)')
    ]:
        m = folium.Map(**map_kwargs)
        add_ee_layer(m, layer_setup[0], layer_setup[1], layer_setup[2])
        if len(layer_setup) > 3:
            add_ee_layer(m, layer_setup[3], layer_setup[4], layer_setup[5])
        maps.append(m)
        
    return maps

# =========================================================================
# 3. STREAMLIT UI & LAYOUT
# =========================================================================

st.markdown("<h3 style='text-align: center; color: red; margin-top: -30px; margin-bottom: 0px;'>UNOFFICIAL</h3>", unsafe_allow_html=True)
st.title("🛰️ Multi-Sensor Intelligence & Monitoring Dashboard")

st.markdown("""
This dashboard provides a unified tactical monitoring system that pairs multi-sensor satellite data with real-time open-source intelligence. By computing automated server-side optical structural anomalies (SSIM) and cleaned radar variations (SAR log-ratio) across dual time frames via Google Earth Engine, it highlights critical physical shifts on the ground alongside filtered geopolitical briefing updates. Ultimately, it serves as a streamlined, low-latency reconnaissance workspace for detecting and contextualizing changes at key global flashpoints.
""")

with st.expander("🌍 The Copernicus Programme & Technologies"):
    st.markdown("""
    ### 🌍 The Copernicus Programme
    * **What it is:** The European Union's flagship Earth observation initiative, often called Europe's "eyes on Earth." 👀
    * **The Goal:** It continuously monitors our planet's environment, oceans, and atmosphere to provide data for climate change tracking, disaster management, and security. 📊
    * **Open Data:** The programme collects vast amounts of global data daily from satellites and ground sensors, making it completely free and open to scientists, businesses, and the public. 🔓

    ### 🛰️ Sentinel-1 Mission
    * **The Radar Specialist:** Sentinel-1 is a polar-orbiting satellite constellation focused entirely on radar imaging. 📡
    * **Day & Night Vision:** Because it uses radar rather than optical cameras, it can capture high-resolution images of Earth's surface through total darkness, heavy rain, and thick cloud cover. ☁️🌙
    * **Primary Uses:** It is primarily used to track Arctic sea ice extent, monitor oil spills, map floods, and detect subtle ground movements caused by earthquakes or landslides. 🌋

    ### 📸 Sentinel-2 Mission
    * **The Optical Photographer:** Sentinel-2 is a constellation of twin satellites equipped with high-resolution multispectral optical cameras. 🗺️
    * **Color and Beyond:** It captures imagery across 13 different light bands, including visible color (red, green, blue) and infrared light, which helps analyze things invisible to the human eye. 🌈
    * **Primary Uses:** It acts as an agricultural powerhouse, widely used to monitor crop health, track global deforestation, map changes in land cover, and monitor inland water bodies like lakes and rivers. 🌲🌾

    ### 📡 What is Synthetic Aperture Radar (SAR)?
    * **Active Sensing:** Unlike standard cameras that rely on sunlight, a SAR instrument acts like a flashlight. It actively transmits its own microwave radio signals down to Earth and measures the echo that bounces back. 🔦
    * **The "Synthetic Trick":** To get crisp, high-resolution images, a radar satellite traditionally needs a massive physical antenna. SAR tricks physics by using the physical forward movement of the satellite itself to "simulate" a much larger antenna (a synthetic aperture), creating incredibly sharp images. 🧠
    """)

# ----------------- LEFT SIDEBAR (Inputs) -----------------
location_query = st.sidebar.text_input("📍 Target Location (City, Base, Coordinates):")

default_lat, default_lon = 0.0000, 0.0000
display_name = "0.0000, 0.0000" 

if location_query:
    lat_res, lon_res = get_coordinates_opencage(location_query)
    if lat_res is not None:
        default_lat, default_lon = lat_res, lon_res
        display_name = location_query.title()
        st.sidebar.success(f"Locked on: {display_name}")
    else:
        st.sidebar.warning("Could not find coordinates. Using default.")

with st.sidebar.form("dashboard_controls"):
    lat_val = st.number_input("Latitude", value=float(default_lat), format="%.6f")
    lon_val = st.number_input("Longitude", value=float(default_lon), format="%.6f")
    
    st.markdown("---")
    d1_val = st.date_input("Baseline Target Date", value=date(2025, 6, 1))
    d2_val = st.date_input("Comparison Target Date", value=date(2026, 6, 1))
    
    st.markdown("---")
    sensitivity_val = st.slider("Change Detection Sensitivity", 0.0, 1.0, 0.5)
    
    submit_button = st.form_submit_button("Execute Satellite Sweep", type="primary", use_container_width=True)

if submit_button:
    st.session_state['generate_maps'] = True

# ----------------- MAIN LAYOUT (Maps vs News) -----------------
col_maps, col_news = st.columns([3, 1], gap="small")

# ---> RIGHT COLUMN: News Feed
with col_news:
    st.markdown("<h4 style='margin-top: 0px; margin-bottom: 10px;'>📡 Intelligence Feed</h4>", unsafe_allow_html=True)
    
    if st.session_state.get('generate_maps', False):
        with st.spinner("Intercepting global feeds..."):
            articles = fetch_intelligence_news(location_query)
            if articles:
                for article in articles:
                    with st.expander(f"📰 {article['title']}", expanded=False):
                        st.write(article.get('description', 'No summary provided.'))
                        st.caption(f"Source: {article['source']['name']} | Date: {article['publishedAt'][:10]}")
                        st.markdown(f"[Read Full Report]({article['url']})")
            else:
                st.info("No relevant intelligence found for this region/query.")
    else:
        st.info("Awaiting satellite sweep execution...")

# ---> LEFT COLUMN: Satellite Maps
with col_maps:
    if st.session_state.get('generate_maps', False):
        start1, end1 = (d1_val - timedelta(45)).strftime('%Y-%m-%d'), (d1_val + timedelta(45)).strftime('%Y-%m-%d')
        start2, end2 = (d2_val - timedelta(45)).strftime('%Y-%m-%d'), (d2_val + timedelta(45)).strftime('%Y-%m-%d')
        
        with st.spinner("Processing Server-Side Imagery & Calculating Topography..."):
            maps = create_maps(lat_val, lon_val, start1, end1, start2, end2, sensitivity_val)
            
        def render_map_card(title, map_obj, subtitle):
            st.markdown(
                f"<div style='margin-bottom: 0px; margin-top: 5px;'>"
                f"<h5 style='margin: 0px; padding: 0px;'>{title}</h5>"
                f"<p style='margin: 0px; padding: 0px; color: #888; font-size: 0.85rem;'>{subtitle}</p>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            map_html = map_obj.get_root().render()
            map_html = map_html.replace("<head>", "<head><style>html, body {width: 100% !important; height: 100% !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important;}</style>")
            components.html(map_html, height=550, scrolling=False)

        map_row1_col1, map_row1_col2 = st.columns(2, gap="small")
        with map_row1_col1:
            render_map_card("1. Optical Baseline", maps[0], "Sentinel-2 | Standard RGB")
        with map_row1_col2:
            render_map_card("2. Optical Comparison", maps[1], "Sentinel-2 | SSIM Overlays (Red)")
            
        map_row2_col1, map_row2_col2 = st.columns(2, gap="small")
        with map_row2_col1:
            render_map_card("3. SAR Baseline", maps[2], "Sentinel-1 | Active Radar")
        with map_row2_col2:
            render_map_card("4. SAR Comparison", maps[3], "Sentinel-1 | Log-Ratio Overlays (Red)")
    else:
        st.info("Input target coordinates or location in the sidebar to begin reconnaissance.")
