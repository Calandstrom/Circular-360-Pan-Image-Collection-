'''
Used to download all of the images. geocode_cache.json is a json file with a name and Latitude and Longitude


'''
import os
import time
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hmac
import hashlib
import base64
import urllib.parse
import getpass


# ========= USER CONFIG =========
API_KEY               = input("Please enter your Google API key: ")
SIGNING_SECRET        = input("Enter URL signing secret (blank to skip signing): ").strip()
BASE_DATA_DIR         = "data"          # Root folder for all images + metadata
GEOCODE_CACHE         = ""
HEADINGS              = [0, 90, 180, 270]
RADIUS_METERS         = 1000		# Change to meet your needs
RADIUS_SPACING_METERS = 50      # Change to meet your needs
FALLBACK_ENABLED      = True
FALLBACK_DISTANCE_METERS = 20
MAX_RETRIES           = 3
RETRY_DELAY           = 2
QUOTA_STOP            = False           # Stop immediately on 403 to save quota
# ================================

os.makedirs(BASE_DATA_DIR, exist_ok=True)

# ── Session with retry ────────────────────────────────────────────────────────
session = requests.Session()
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ── Quota tracking ────────────────────────────────────────────────────────────
quota_tracker = {
    'requests_today' : 0,
    'errors_403'     : 0,
    'start_time'     : time.time()
}



def sign_url(input_url, secret):
    """Append a Google-compatible HMAC-SHA1 signature to a URL."""
    if not secret:
        return input_url

    parsed = urllib.parse.urlparse(input_url)
    url_to_sign = f"{parsed.path}?{parsed.query}".encode("utf-8")

    # Secret is URL-safe base64; restore padding if it was stripped
    padded = secret + "=" * (-len(secret) % 4)
    decoded_key = base64.urlsafe_b64decode(padded)

    digest = hmac.new(decoded_key, url_to_sign, hashlib.sha1).digest()
    signature = base64.urlsafe_b64encode(digest).decode("utf-8")

    return f"{input_url}&signature={signature}"



# ══════════════════════════════════════════════════════════════════════
# CACHE LOADER
# ══════════════════════════════════════════════════════════════════════
def cache_converter(filepath):
    centers    = {}
    failed_pcn = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f'Cache loaded: {len(cache)} entries')
    except FileNotFoundError:
        print('Cache file not found')
        return centers, failed_pcn
    except json.JSONDecodeError as e:
        print(f'Cache JSON error: {e}')
        return centers, failed_pcn

    for key, value in cache.items():
        lat = lon = None

        if isinstance(value, dict):
            for k in ('Latitude', 'latitude', 'lat'):
                if k in value:
                    lat = value[k]
                    break
            for k in ('Longitude', 'longitude', 'lng', 'lon'):
                if k in value:
                    lon = value[k]
                    break
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            lat, lon = value[0], value[1]

        if lat is not None and lon is not None:
            try:
                centers[key] = [float(lat), float(lon)]
            except (TypeError, ValueError):
                failed_pcn.append(key)
                print(f'  Skipping {key}: invalid lat/lon')
        else:
            failed_pcn.append(key)
            print(f'  Skipping {key}: missing lat/lon')

    print(f'Valid PCNs loaded: {len(centers)}')
    if failed_pcn:
        print(f'Failed to load  : {len(failed_pcn)}')

    return centers, failed_pcn

# ══════════════════════════════════════════════════════════════════════
# POINT GENERATION
# ══════════════════════════════════════════════════════════════════════
def meters_to_degrees(lat, meters):
    lat_deg = meters / 111320.0
    lon_deg = meters / (111320.0 * abs(np.cos(np.radians(lat))))
    return lat_deg, lon_deg

def generate_radius_points(center_lat, center_lon, radius_meters, spacing_meters):
    points = []
    lat_step, lon_step = meters_to_degrees(center_lat, spacing_meters)
    rad_lat, rad_lon   = meters_to_degrees(center_lat, radius_meters)

    lat = center_lat - rad_lat
    while lat <= center_lat + rad_lat:
        lon = center_lon - rad_lon
        while lon <= center_lon + rad_lon:
            dist_lat = (lat - center_lat) * 111320.0
            dist_lon = (lon - center_lon) * 111320.0 * abs(np.cos(np.radians(center_lat)))
            if np.sqrt(dist_lat**2 + dist_lon**2) <= radius_meters:
                points.append({"lat": lat, "lon": lon})
            lon += lon_step
        lat += lat_step

    return points

# ══════════════════════════════════════════════════════════════════════
# FALLBACK SYSTEM
# ══════════════════════════════════════════════════════════════════════
def get_fallback_points(lat, lon, distance_meters):
    lat_step, lon_step = meters_to_degrees(lat, distance_meters)
    return [
        {"lat": lat + lat_step, "lon": lon,          "direction": "N"},
        {"lat": lat,            "lon": lon + lon_step,"direction": "E"},
        {"lat": lat - lat_step, "lon": lon,           "direction": "S"},
        {"lat": lat,            "lon": lon - lon_step,"direction": "W"},
    ]

def get_metadata_for_point(lat, lon, api_key):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&key={api_key}"
    url = sign_url(url, SIGNING_SECRET) 
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
            elif r.status_code == 403:
                return {"status": r.text[:500]}
            else:
                return {"status": f"HTTP_ERROR_{r.status_code}"}
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  Connection error. Retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                return {"status": "CONNECTION_ERROR", "error": str(e)}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    return {"status": "MAX_RETRIES_EXCEEDED"}

def get_point_with_fallback(req_lat, req_lon, api_key):
    meta = get_metadata_for_point(req_lat, req_lon, api_key)

    if meta.get("status") == "OK":
        return meta, req_lat, req_lon, None

    if FALLBACK_ENABLED and meta.get("status") not in ("error_403", "CONNECTION_ERROR"):
        for fallback in get_fallback_points(req_lat, req_lon, FALLBACK_DISTANCE_METERS):
            fb_meta = get_metadata_for_point(fallback["lat"], fallback["lon"], api_key)
            if fb_meta.get("status") == "OK":
                return fb_meta, fallback["lat"], fallback["lon"], fallback["direction"]
            time.sleep(0.1)

    return meta, req_lat, req_lon, None

# ══════════════════════════════════════════════════════════════════════
# IMAGE DOWNLOAD
# ══════════════════════════════════════════════════════════════════════
def download_image_for_heading(pano_id, heading, api_key, out_path):
    url = (f"https://maps.googleapis.com/maps/api/streetview"
           f"?size=640x640&pano={pano_id}&heading={heading}&pitch=0&key={api_key}")
    url = sign_url(url, SIGNING_SECRET) 
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, stream=True, timeout=20)

            if r.status_code == 200:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                quota_tracker['requests_today'] += 1
                return True, 200

            elif r.status_code == 403:
                # ── Quota exceeded — stop everything ─────────────────────────
                quota_tracker['errors_403'] += 1
                elapsed = (time.time() - quota_tracker['start_time']) / 3600
                print(f"\n{'='*60}")
                print(f"QUOTA EXCEEDED (403) — stopping to preserve quota")
                print(f"  Requests this session : {quota_tracker['requests_today']}")
                print(f"  Session duration      : {elapsed:.1f} hours")
                print(f"  403 errors seen       : {quota_tracker['errors_403']}")
                print(f"  Progress is saved — rerun tomorrow to resume")
                print(f"  Quota resets at midnight Pacific Time")
                print(f"  To remove limit: console.cloud.google.com")
                print(f"  → APIs & Services → Street View Static API → Quotas")
                print(f"{'='*60}\n")
                if QUOTA_STOP:
                    raise SystemExit("Stopping due to 403 quota error")
                return False, 403

            elif r.status_code == 429:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)

            else:
                if attempt == MAX_RETRIES - 1:
                    return False, r.status_code

        except SystemExit:
            raise
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  Connection error. Retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                return False, f"CONNECTION_ERROR"
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return False, f"ERROR: {str(e)}"

    return False, "MAX_RETRIES_EXCEEDED"

def format_coord(coord):
    return f"{coord:.6f}".replace(".", "_").replace("-", "neg")

# ══════════════════════════════════════════════════════════════════════
# SKIP LOGIC — only download folders that need it
# ══════════════════════════════════════════════════════════════════════
def should_skip_pcn(pcn_name, base_data_dir):
    """
    Returns (skip, reason) tuple.
    Skips if folder has images AND metadata has no error_403 rows.
    Redownloads if folder is empty OR has error_403 rows.
    """
    safe_name     = pcn_name.replace(' ', '_')
    folder_path   = os.path.join(base_data_dir, f"images_{safe_name}")
    metadata_path = os.path.join(base_data_dir, f"images_metadata_{safe_name}.parquet")

    # No folder at all — needs full download
    if not os.path.exists(folder_path):
        return False, "no folder"

    # Folder exists but check metadata
    if os.path.exists(metadata_path):
        try:
            meta      = gpd.read_parquet(metadata_path)
            ok_count  = (meta['status'] == 'ok').sum()
            err_count = (meta['status'] == 'error_403').sum()

            if ok_count > 0 and err_count == 0:
                return True, f"complete ({ok_count} images)"
            elif err_count > 0:
                return False, f"has {err_count} error_403 rows to retry"
            else:
                return False, "no ok images"
        except Exception as e:
            return False, f"metadata read error: {e}"

    # Folder exists but no metadata — check image count
    n_images = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    if n_images > 0:
        return True, f"folder has {n_images} images (no metadata)"

    return False, "folder is empty"

# ══════════════════════════════════════════════════════════════════════
# MAIN DOWNLOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════
def download_pcn(pcn_name, lat, lon, api_key, base_data_dir):
    safe_name     = pcn_name.replace(' ', '_')
    output_dir    = os.path.join(base_data_dir, f"images_{safe_name}")
    metadata_path = os.path.join(base_data_dir, f"images_metadata_{safe_name}.parquet")

    os.makedirs(output_dir, exist_ok=True)

    # Load existing metadata if retrying 403s
    existing_ok_rows = []
    if os.path.exists(metadata_path):
        try:
            existing_meta    = gpd.read_parquet(metadata_path)
            existing_ok_rows = existing_meta[
                existing_meta['status'] == 'ok'
            ].to_dict('records')
        except Exception:
            existing_ok_rows = []

    points = generate_radius_points(lat, lon, RADIUS_METERS, RADIUS_SPACING_METERS)
    print(f"  {pcn_name}: {len(points)} points to process")

    rows              = list(existing_ok_rows)  # keep existing ok rows
    successful_points = 0
    failed_points     = 0
    skipped_points    = 0

    # Build set of already-downloaded file paths to avoid re-downloading
    existing_files = set(
        f for f in os.listdir(output_dir)
        if f.endswith('.jpg')
    )

    for point in tqdm(points, desc=f"  {pcn_name}", leave=False):
        req_lat = point["lat"]
        req_lon = point["lon"]

        meta, snap_lat, snap_lon, fallback_dir = get_point_with_fallback(
            req_lat, req_lon, api_key
        )

        if meta.get("status") != "OK":
            failed_points += 1
            rows.append({
                "pano_id"      : None,
                "requested_lat": req_lat,
                "requested_lon": req_lon,
                "snap_lat"     : snap_lat,
                "snap_lon"     : snap_lon,
                "heading"      : None,
                "capture_date" : None,
                "file_path"    : None,
                "status"       : meta.get("status", "UNKNOWN"),
                "fallback_used": fallback_dir
            })
            continue

        successful_points += 1
        pano_id      = meta.get("pano_id") or meta.get("panoId") or meta.get("panoid")
        loc_obj      = meta.get("location", {})
        snap_lat     = loc_obj.get("lat", snap_lat)
        snap_lon     = loc_obj.get("lng", snap_lon)
        capture_date = meta.get("date", None)

        for heading in HEADINGS:
            lat_str  = format_coord(snap_lat)
            lon_str  = format_coord(snap_lon)
            filename = f"{lat_str}_{lon_str}_{heading}.jpg"

            # ── Skip if already downloaded successfully ────────────────────
            if filename in existing_files:
                skipped_points += 1
                rows.append({
                    "pano_id"      : pano_id,
                    "requested_lat": req_lat,
                    "requested_lon": req_lon,
                    "snap_lat"     : snap_lat,
                    "snap_lon"     : snap_lon,
                    "heading"      : heading,
                    "capture_date" : capture_date,
                    "file_path"    : os.path.join('data', f"images_{safe_name}", filename),
                    "status"       : "ok",
                    "fallback_used": fallback_dir
                })
                continue

            # ── Download ───────────────────────────────────────────────────
            out_path = os.path.join(output_dir, filename)
            ok, code = download_image_for_heading(pano_id, heading, api_key, out_path)

            rows.append({
                "pano_id"      : pano_id,
                "requested_lat": req_lat,
                "requested_lon": req_lon,
                "snap_lat"     : snap_lat,
                "snap_lon"     : snap_lon,
                "heading"      : heading,
                "capture_date" : capture_date,
                # ── Store as relative path from imageCollection root ───────
                "file_path"    : os.path.join('data', f"images_{safe_name}", filename),
                "status"       : "ok" if ok else f"error_{code}",
                "fallback_used": fallback_dir
            })
            time.sleep(0.1)

    # ── Save metadata ─────────────────────────────────────────────────────────
    if rows:
        df  = pd.DataFrame(rows)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df["snap_lon"].fillna(0),
                df["snap_lat"].fillna(0)
            ),
            crs="EPSG:4326"
        )
        gdf.to_parquet(metadata_path)

    ok_count  = sum(1 for r in rows if r.get('status') == 'ok')
    err_count = sum(1 for r in rows if str(r.get('status', '')).startswith('error'))

    print(f"  ✓ {pcn_name}: {ok_count} ok | {err_count} errors | {skipped_points} skipped")
    return ok_count

# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    centers, failed = cache_converter(GEOCODE_CACHE)

    if not centers:
        print("No valid PCNs in cache. Exiting.")
        exit(1)

    # ── Categorise PCNs ───────────────────────────────────────────────────────
    to_download = []
    to_skip     = []

    for pcn_name, coords in centers.items():
        skip, reason = should_skip_pcn(pcn_name, BASE_DATA_DIR)
        if skip:
            to_skip.append((pcn_name, reason))
        else:
            to_download.append((pcn_name, coords, reason))

    print(f"\n{'='*60}")
    print(f"Download plan:")
    print(f"  Total PCNs in cache : {len(centers)}")
    print(f"  Skipping (complete) : {len(to_skip)}")
    print(f"  Downloading         : {len(to_download)}")
    print(f"  Estimated API calls : {len(to_download) * 300 * 4:,}")
    print(f"  Days needed (free)  : {len(to_download) * 300 * 4 / 25000:.1f}")
    print(f"{'='*60}\n")

    # ── Download ──────────────────────────────────────────────────────────────
    completed = 0
    for pcn_name, coords, reason in to_download:
        lat, lon = coords[0], coords[1]
        print(f"\n[{completed+1}/{len(to_download)}] {pcn_name} ({reason})")
        download_pcn(pcn_name, lat, lon, API_KEY, BASE_DATA_DIR)
        completed += 1

    print(f"\n{'='*60}")
    print(f"Session complete:")
    print(f"  PCNs processed      : {completed}")
    print(f"  PCNs skipped        : {len(to_skip)}")
    print(f"  Total API requests  : {quota_tracker['requests_today']}")
    print(f"  403 errors seen     : {quota_tracker['errors_403']}")
    print(f"{'='*60}")
