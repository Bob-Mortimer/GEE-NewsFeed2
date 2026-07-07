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
st.set_page_config(layout="wide", page_title="Geospatial Intelligence Dashboard")

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
# 2. CORE INTELLIGENCE LOGIC (NewsAPI, OpenCage, Earth Engine)
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
    """Fetches news optimized into a single API call to preserve rate limits."""
    api_key = st.secrets.get("newsapi_key", None)
    if not api_key:
        st.error("Missing NewsAPI Key in secrets.")
        return []
        
    # Core geopolitical/military keywords
    keywords = '(military OR cyber OR politics OR political OR president OR "prime minister" OR conflict OR flashpoint)'
    
    # Securely bundle user query with core intelligence keywords
    q = f"({keywords}) AND ({user_query})" if user_query else keywords
    
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
    """Safely extracts map tiles from GEE and embeds them onto a Folium layer"""
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
    """Generates the 4 analytical Folium maps calculated server-side in GEE."""
    point = ee.Geometry.Point([lon, lat])
    
    # -------------------------------------------------------------
    # OPTICAL: Sentinel-2 Data
    # -------------------------------------------------------------
    s2_base = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
               .filterBounds(point).filterDate(start1, end1)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
               .map(mask_s2_clouds).median())
    
    s2_comp = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
               .filterBounds(point).filterDate(start2, end2)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
               .map(mask_s2_clouds).median())
               
    # ADVANCED METHOD: Structural Similarity Index (SSIM) on Red Band (B4)
    b1 = s2_base.select('B4')
    b2 = s2_comp.select('B4')
    
    # SSIM Constants & Kernel
    L = 10000 
    c1 = (0.01 * L)**2
    c2 = (0.03 * L)**2
    kernel = ee.Kernel.square(3) # 3x3 neighborhood window
    
    mu1 = b1.reduceNeighborhood(ee.Reducer.mean(), kernel)
    mu2 = b2.reduceNeighborhood(ee.Reducer.mean(), kernel)
    var1 = b1.reduceNeighborhood(ee.Reducer.variance(), kernel)
    var2 = b2.reduceNeighborhood(ee.Reducer.variance(), kernel)
    
    # Covariance approximation E[x*y] - E[x]*E[y]
    mu_1_2 = b1.multiply(b2).reduceNeighborhood(ee.Reducer.mean(), kernel)
    cov12 = mu_1_2.subtract(mu1.multiply(mu2))
    
    # SSIM Equation
    num = (mu1.multiply(mu2).multiply(2).add(c1)).multiply(cov12.multiply(2).add(c2))
    den = (mu1.pow(2).add(mu2.pow(2)).add(c1)).multiply(var1.add(var2).add(c2))
    ssim = num.divide(den)
    
    # Calculate difference mask based on sensitivity (Lower SSIM = higher structural change)
    ssim_threshold = 1.0 - (sensitivity * 0.5) 
    ssim_mask = ssim.lt(ssim_threshold)
    s2_diff_red_dots = ssim_mask.updateMask(ssim_mask)
    
    # -------------------------------------------------------------
    # RADAR: Sentinel-1 SAR Data
    # -------------------------------------------------------------
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

    # ADVANCED METHOD: SAR Log-Ratio with Morphological Cleaning
    # S1 is natively in decibels (dB), meaning subtraction effectively calculates the log-ratio.
    s1_diff_abs = s1_comp.subtract(s1_base).abs()
    
    # Map sensitivity (0.0 to 1.0) to a dB threshold. Higher sensitivity = lower threshold
    db_threshold = 4.5 - (sensitivity * 3) 
    s1_raw_mask = s1_diff_abs.gt(db_threshold)
    
    # Morphological Cleaning: Use focal mode to erase random speckle noise
    s1_clean_mask = s1_raw_mask.focal_mode(radius=30, units='meters')
    s1_red_dots = s1_clean_mask.updateMask(s1_clean_mask)

    # -------------------------------------------------------------
    # BUILD MAPS
    # -------------------------------------------------------------
    vis_s2 = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
    vis_diff = {'min': 1, 'max': 1, 'palette': ['FF0000']} 
    vis_s1 = {'min': -25, 'max': 0}
    
    maps = []
    # Map 1: S2 Baseline
    m1 = folium.Map(location=[lat, lon], zoom_start=13, tiles='CartoDB dark_matter', height=400)
    add_ee_layer(m1, s2_base, vis_s2, 'S2 Baseline')
    maps.append(m1)

    # Map 2: S2 Comparison WITH SSIM Differences overlay
    m2 = folium.Map(location=[lat, lon], zoom_start=13, tiles='CartoDB dark_matter', height=400)
    add_ee_layer(m2, s2_comp, vis_s2, 'S2 Comparison')
    add_ee_layer(m2, s2_diff_red_dots, vis_diff, 'Optical Change (SSIM)')
    maps.append(m2)

    # Map 3: S1 Baseline
    m3 = folium.Map(location=[lat, lon], zoom_start=13, tiles='CartoDB dark_matter', height=400)
    add_ee_layer(m3, s1_base, vis_s1, 'S1 Baseline')
    maps.append(m3)

    # Map 4: S1 Comparison WITH SAR Differences overlay
    m4 = folium.Map(location=[lat, lon], zoom_start=13, tiles='CartoDB dark_matter', height=400)
    add_ee_layer(m4, s1_comp, vis_s1, 'S1 Comparison')
    add_ee_layer(m4, s1_red_dots, vis_diff, 'SAR Change (Cleaned)')
    maps.append(m4)
        
    return maps

# =========================================================================
# 3. STREAMLIT UI & LAYOUT
# =========================================================================

st.markdown("<h3 style='text-align: center; color: red; margin-bottom: 0px;'>UNOFFICIAL</h3>", unsafe_allow_html=True)
st.title("🛰️ Multi-Sensor Intelligence & Monitoring Dashboard")

# ----------------- LEFT SIDEBAR (Inputs) -----------------
location_query = st.sidebar.text_input("📍 Target Location (City, Base, Coordinates):")

default_lat, default_lon = -35.2809, 149.1300
display_name = "Canberra, Australia" 

if location_query:
    lat_res, lon_res = get_coordinates_opencage(location_query)
    if lat_res:
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
col_maps, col_news = st.columns([3, 1])

# ---> RIGHT COLUMN: News Feed
with col_news:
    st.subheader("📡 Intelligence Feed")
    news_search = st.text_input("Filter topics (Optional):", placeholder="e.g. 'nuclear' or 'naval base'")
    
    if st.button("Fetch Briefings", use_container_width=True):
        with st.spinner("Intercepting feeds..."):
            articles = fetch_intelligence_news(news_search)
            if articles:
                for article in articles:
                    with st.expander(f"📰 {article['title']}", expanded=False):
                        st.write(article.get('description', 'No summary provided.'))
                        st.caption(f"Source: {article['source']['name']} | Date: {article['publishedAt'][:10]}")
                        st.markdown(f"[Read Full Report]({article['url']})")
            else:
                st.info("No relevant intelligence found for this query.")

# ---> LEFT COLUMN: Satellite Maps
with col_maps:
    st.subheader(f"Target: {display_name}")
    
    if st.session_state.get('generate_maps', False):
        start1, end1 = (d1_val - timedelta(45)).strftime('%Y-%m-%d'), (d1_val + timedelta(45)).strftime('%Y-%m-%d')
        start2, end2 = (d2_val - timedelta(45)).strftime('%Y-%m-%d'), (d2_val + timedelta(45)).strftime('%Y-%m-%d')
        
        with st.spinner("Processing Server-Side SSIM and SAR Log-Ratios via GEE..."):
            maps = create_maps(lat_val, lon_val, start1, end1, start2, end2, sensitivity_val)
            
        def render_map_card(title, map_obj, subtitle):
            st.markdown(f"<h5 style='margin-bottom: 2px; margin-top: 5px;'>{title}</h5>", unsafe_allow_html=True)
            st.caption(subtitle)
            components.html(map_obj._repr_html_(), height=410, scrolling=False)

        # 2x2 Grid inside the Map Column
        map_row1_col1, map_row1_col2 = st.columns(2)
        with map_row1_col1:
            render_map_card("1. Optical Baseline", maps[0], "Sentinel-2 | Standard RGB")
        with map_row1_col2:
            render_map_card("2. Optical Comparison", maps[1], "Sentinel-2 | SSIM Overlays (Red)")
            
        st.write("---")
        
        map_row2_col1, map_row2_col2 = st.columns(2)
        with map_row2_col1:
            render_map_card("3. SAR Baseline", maps[2], "Sentinel-1 | Active Radar")
        with map_row2_col2:
            render_map_card("4. SAR Comparison", maps[3], "Sentinel-1 | Log-Ratio Overlays (Red)")
