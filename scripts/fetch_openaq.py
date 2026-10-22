"""
Download hourly PM2.5 for every OpenAQ sensor near Accra.

    python scripts/fetch_openaq.py                 # all active sensors, full history
    python scripts/fetch_openaq.py --since 2026-10-01   # only recent hours (for updates)

Needs OPENAQ_API_KEY in .env. Writes to data/raw/openaq_accra/ (gitignored):
  sensors.csv      one row per sensor: id, name, location, provider, coordinates, first/last report
  pm25_hourly.csv  sensor_id, hour (UTC, start of the hour), pm25 (µg/m³), coverage (% of the hour observed)
Re-running resumes: sensors already fully downloaded are skipped unless --refresh.
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

API = "https://api.openaq.org/v3"
ACCRA = (5.6037, -0.1870)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "openaq_accra"
REQUEST_GAP_S = 1.1  # free keys allow 60 requests/minute


def api_key() -> str:
    """
    Returns:
        The OpenAQ API key from the environment or .env.

    Raises:
        SystemExit: if OPENAQ_API_KEY isn't set.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("OPENAQ_API_KEY missing: add it to .env")
    return key


def get(path, params, key):
    """
    One GET request to the OpenAQ v3 API, retried with growing waits on rate
    limits (429), server errors (5xx) and network failures. It pauses
    REQUEST_GAP_S after each success to stay under 60 requests/minute.

    Args:
        path: API path, e.g. "/locations" or "/sensors/123/hours".
        params: dict of query parameters.
        key: API key.

    Returns:
        The decoded JSON response (a dict with "meta" and "results").

    Raises:
        urllib.error.HTTPError: for other HTTP errors (e.g. 401 bad key, 404).
        RuntimeError: after 6 failed attempts.
    """
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"X-API-Key": key})
            with urllib.request.urlopen(req, timeout=120) as r:
                time.sleep(REQUEST_GAP_S)
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(10 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def pm25_sensors(key, radius_m, active_since):
    """
    List the PM2.5 sensors near Accra.

    Args:
        key: API key.
        radius_m: search radius around central Accra, in metres (API maximum 25000).
        active_since: ISO date. Only locations that reported on or after it are kept.

    Returns:
        list of dicts: sensor_id, location_id, name, provider, lat, lon, first, last
        (the first and last report times in UTC).
    """
    locs = get("/locations", {"coordinates": f"{ACCRA[0]},{ACCRA[1]}", "radius": radius_m, "limit": 1000}, key)["results"]
    rows = []
    for loc in locs:
        last = (loc.get("datetimeLast") or {}).get("utc") or ""
        for s in loc.get("sensors", []):
            if s["parameter"]["name"] == "pm25" and last >= active_since:
                rows.append({
                    "sensor_id": s["id"], "location_id": loc["id"], "name": (loc.get("name") or "").strip(),
                    "provider": (loc.get("provider") or {}).get("name", ""),
                    "lat": loc["coordinates"]["latitude"], "lon": loc["coordinates"]["longitude"],
                    "first": (loc.get("datetimeFirst") or {}).get("utc", ""), "last": last,
                })
    return rows


def sensor_hours(key, sensor_id, date_from, date_to, chunk_days=90):
    """
    All hourly PM2.5 values of one sensor between two dates. Requested in
    `chunk_days` slices because the API fails on multi-year ranges, and page by
    page (1000 hours per page) within each slice.

    Args:
        key: API key.
        sensor_id: OpenAQ sensor id.
        date_from: start, ISO date or datetime (UTC).
        date_to: end, ISO date or datetime (UTC).
        chunk_days: slice length in days.

    Yields:
        (hour, pm25, coverage) tuples: hour = start of the hour (UTC, ISO text),
        pm25 = hourly average in µg/m³, coverage = OpenAQ's percent-complete for that hour.
    """
    start = datetime.fromisoformat(date_from.replace("Z", "")[:19])
    end = datetime.fromisoformat(date_to.replace("Z", "")[:19])
    while start < end:
        stop = min(start + timedelta(days=chunk_days), end)
        page = 1
        while True:
            res = get(f"/sensors/{sensor_id}/hours", {"datetime_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                      "datetime_to": stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                      "limit": 1000, "page": page}, key)["results"]
            for x in res:
                yield (x["period"]["datetimeFrom"]["utc"], x["value"], (x.get("coverage") or {}).get("percentComplete"))
            if len(res) < 1000:
                break
            page += 1
        start = stop


def main():
    """
    Download the sensor list and every sensor's hourly history (see the module
    docstring for the options and output files). Each sensor's rows are written
    only after it has fully downloaded, so an interrupted run never leaves half a
    sensor in the file, and a re-run picks up where it stopped.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=25000, help="metres around central Accra (API max 25000)")
    ap.add_argument("--active-since", default="2026-09-01", help="only sensors that reported on/after this date")
    ap.add_argument("--since", default=None, help="only download hours from this date (default: each sensor's first report)")
    ap.add_argument("--until", default=time.strftime("%Y-%m-%dT%H:00:00Z", time.gmtime()))
    ap.add_argument("--refresh", action="store_true", help="re-download sensors already in pm25_hourly.csv")
    args = ap.parse_args()

    key = api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    sensors = pm25_sensors(key, args.radius, args.active_since)
    with open(OUT / "sensors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sensors[0]))
        w.writeheader()
        w.writerows(sensors)
    print(f"{len(sensors)} PM2.5 sensors -> {OUT / 'sensors.csv'}")

    hourly = OUT / "pm25_hourly.csv"
    done = set()
    if hourly.exists() and not args.refresh and args.since is None:
        with open(hourly, encoding="utf-8") as f:
            done = {row["sensor_id"] for row in csv.DictReader(f)}
    mode = "a" if done else "w"
    with open(hourly, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["sensor_id", "hour", "pm25", "coverage"])
        for i, s in enumerate(sensors, 1):
            if str(s["sensor_id"]) in done:
                continue
            start = args.since or s["first"][:10]
            # written only once the whole sensor is in, so an interrupted run never leaves half a sensor
            rows = [[s["sensor_id"], hour, value, cov]
                    for hour, value, cov in sensor_hours(key, s["sensor_id"], start, args.until)]
            w.writerows(rows)
            f.flush()
            print(f"[{i}/{len(sensors)}] {s['name'][:40]:40s} {len(rows):6d} hours", flush=True)


if __name__ == "__main__":
    main()
