"""Photographic chat backgrounds, fetched once and served locally.

Hotlinking would mean a request to a third party every time the dashboard loads,
which tells them your IP and breaks when you are offline. Downloading once keeps
Bolt self-contained, which is the whole point of running it on your own machine.

Source: Unsplash. Its licence allows free use, commercially or not, without
permission or attribution — though attribution is appreciated, so the manifest
records the photographer and the source URL for every image.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from bolt.core import workspace

BACKGROUND_DIR = Path(os.environ.get(
    "BOLT_BACKGROUND_DIR", workspace.DEFAULT / "backgrounds"))
MANIFEST = BACKGROUND_DIR / "manifest.json"
WIDTH = int(os.environ.get("BOLT_BACKGROUND_WIDTH", "1920"))
QUALITY = int(os.environ.get("BOLT_BACKGROUND_QUALITY", "78"))
# A remote server decides how many bytes it sends. Stream, and stop early rather
# than hold whatever arrives in memory.
MAX_PHOTO_BYTES = int(os.environ.get("BOLT_BACKGROUND_MAX_MB", "25")) * 1_000_000
LICENCE = ("Unsplash License — free to use, commercial or not, no permission "
           "required. https://unsplash.com/license")


@dataclass(frozen=True)
class Photo:
    id: str          # our name for it
    name: str        # what a person sees
    category: str
    photo: str       # the Unsplash photo id


# Curated rather than random: a background is looked at for hours, so it wants to
# be calm and largely empty in the middle where the conversation sits.
CATALOGUE: list[Photo] = [
    # --- earth and sky ---
    Photo("earth",        "Earth",          "Earth",  "1451187580459-43490279c0fa"),
    Photo("space",        "Deep space",     "Earth",  "1446776877081-d282a0f896e2"),
    Photo("nebula",       "Nebula",         "Earth",  "1502134249126-9f3755a50d78"),
    Photo("mountains",    "Mountains",      "Earth",  "1519681393784-d120267933ba"),
    Photo("forest-fog",   "Forest fog",     "Earth",  "1470071459604-3b5ec3a7fe05"),
    Photo("forest",       "Pines",          "Earth",  "1441974231531-c6227db76b6e"),
    Photo("lake",         "Still lake",     "Earth",  "1470252649378-9c29740c9fa8"),
    Photo("mountain-lake", "Mountain lake", "Earth",  "1506905925346-21bda4d32df4"),
    Photo("waterfall",    "Waterfall",      "Earth",  "1433086966358-54859d0ed716"),
    Photo("valley",       "Valley",         "Earth",  "1501785888041-af3ef285b470"),
    Photo("ocean",        "Ocean",          "Earth",  "1518837695005-2083093ee35b"),
    Photo("beach",        "Shore",          "Earth",  "1507525428034-b723cf961d3e"),
    Photo("alps",         "Alps",           "Earth",  "1497436072909-60f360e1d4b1"),
    Photo("field",        "Open field",     "Earth",  "1472214103451-9374bd1c798e"),
    Photo("desert",       "Dunes",          "Earth",  "1518998053901-5348d3961a04"),
    Photo("aurora-sky",   "Aurora",         "Earth",  "1483347756197-71ef80e95f73"),
    Photo("night-sky",    "Night sky",      "Earth",  "1475274047050-1d0c0975c63e"),
    Photo("canyon",       "Canyon",         "Earth",  "1474044159687-1ee9f3a51722"),
    Photo("river",        "River",          "Earth",  "1439853949127-fa647821eba0"),
    Photo("snow",         "Snowfield",      "Earth",  "1418985991508-e47386d96a71"),

    # --- built things ---
    Photo("building",     "Concrete",       "Architecture", "1487958449943-2429e8be8625"),
    Photo("city",         "City",           "Architecture", "1449824913935-59a10b8d2000"),
    Photo("skyline",      "Skyline",        "Architecture", "1477959858617-67f85cf4f1df"),
    Photo("modern",       "Modern",         "Architecture", "1518005020951-eccb494ad742"),
    Photo("stairs",       "Stairwell",      "Architecture", "1470723710355-95304d8aece4"),
    Photo("bridge",       "Bridge",         "Architecture", "1509023464722-18d996393ca8"),
    Photo("atrium",       "Atrium",         "Architecture", "1481253127861-534498168948"),
    Photo("facade",       "Facade",         "Architecture", "1486406146926-c627a92ad1ab"),
    Photo("library",      "Library",        "Architecture", "1507842217343-583bb7270b66"),
    Photo("arches",       "Arches",         "Architecture", "1520962880247-cfaf541c8724"),

    # --- code and desks ---
    Photo("code",         "Code",           "Code", "1461749280684-dccba630e2f6"),
    Photo("code-dark",    "Terminal",       "Code", "1555066931-4365d14bab8c"),
    Photo("laptop",       "Laptop",         "Code", "1517180102446-f3ece451e9d8"),
    Photo("desk",         "Desk",           "Code", "1498050108023-c5249f4df085"),
    Photo("workspace",    "Workspace",      "Code", "1531297484001-80022131f5a1"),
    Photo("circuit",      "Circuitry",      "Code", "1518770660439-4636190af475"),
    Photo("server",       "Server room",    "Code", "1558494949-ef010cbdcc31"),
    Photo("editor",       "Editor",         "Code", "1607799279861-4dd421887fb3"),
    Photo("syntax",       "Syntax",         "Code", "1542831371-29b0f74f9713"),
    Photo("glasses",      "Glasses",        "Code", "1504639725590-34d0984388bd"),
    Photo("standing-desk", "Standing desk", "Code", "1550439062-609e1531270e"),
    Photo("matrix",       "Green rain",     "Code", "1526374965328-7f61d4dc18c5"),
    Photo("keyboard",     "Backlit keys",   "Code", "1571171637578-41bc2dd41cd2"),
    Photo("desk-plant",   "Desk plant",     "Code", "1517694712202-14dd9538aa97"),
    Photo("datacenter",   "Data centre",    "Code", "1573164713988-8665fc963095"),
    Photo("devices",      "Devices",        "Code", "1519389950473-47ba0277781c"),
    Photo("notebook",     "Notebook",       "Code", "1484417894907-623942c8ee29"),
    Photo("editor-wide",  "Wide editor",    "Code", "1516116216624-53e697fedbea"),
    Photo("laptop-code",  "Laptop code",    "Code", "1587620962725-abab7fe55159"),
    Photo("dark-screen",  "Dark screen",    "Code", "1509966756634-9c23dd6e6815"),
    Photo("bokeh-code",   "Code bokeh",     "Code", "1555949963-ff9fe0c870eb"),
    Photo("cables",       "Patch panel",    "Code", "1544197150-b99a580bb7a8"),
    Photo("typing",       "Typing",         "Code", "1486312338219-ce68d2c6f44d"),
    Photo("flatlay",      "Flat lay",       "Code", "1522252234503-e356532cafd5"),

    # --- cars ---
    Photo("car-highway",  "Highway",        "Cars", "1503376780353-7e6692767b70"),
    Photo("car-camaro",   "Blue muscle",    "Cars", "1552519507-da3b142c6e3d"),
    Photo("car-mustang",  "Mustang",        "Cars", "1494976388531-d1058494cdd8"),
    Photo("car-ferrari",  "Red supercar",   "Cars", "1583121274602-3e2820c69888"),
    Photo("car-coupe",    "Blue coupe",     "Cars", "1502877338535-766e1452684a"),
    Photo("car-hatch",    "Teal hatchback", "Cars", "1541899481282-d53bffe3c35d"),
    Photo("car-desert",   "Desert road",    "Cars", "1533473359331-0135ef1b58bf"),
    Photo("car-hyper",    "Hypercar",       "Cars", "1544636331-e26879cd4d9b"),
    Photo("car-studio",   "Studio grey",    "Cars", "1580273916550-e323be2ae537"),
    Photo("car-stripe",   "Yellow stripe",  "Cars", "1514316454349-750a7fd3da3a"),
    Photo("car-sedan",    "Sport sedan",    "Cars", "1555215695-3004980ad54e"),
    Photo("car-city",     "City runabout",  "Cars", "1549317661-bd32c8ce0db2"),
    Photo("car-garage",   "Garage",         "Cars", "1492144534655-ae79c964c9d7"),
    Photo("car-forest",   "Forest road",    "Cars", "1553440569-bcc63803a83d"),
    Photo("car-yellow",   "Yellow wedge",   "Cars", "1511919884226-fd3cad34687c"),
    Photo("car-avenue",   "Tree avenue",    "Cars", "1517524008697-84bbe3c3fd98"),

    # --- quiet life ---
    Photo("coffee",       "Coffee",         "Life", "1495474472287-4d71bcdd2085"),
    Photo("books",        "Books",          "Life", "1512820790803-83ca734da794"),
    Photo("plants",       "Plants",         "Life", "1466692476868-aef1dfb1e735"),
    Photo("window",       "Rain window",    "Life", "1428592953211-077101b2021b"),
    Photo("linen",        "Linen",          "Life", "1493663284031-b7e3aefcae8e"),
]


def url_for(photo: Photo, width: int = WIDTH) -> str:
    return (f"https://images.unsplash.com/photo-{photo.photo}"
            f"?w={width}&q={QUALITY}&fm=jpg&fit=crop")


def source_page(photo: Photo) -> str:
    return f"https://unsplash.com/photos/{photo.photo.split('-')[-1]}"


def installed() -> dict[str, dict]:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8")).get("photos", {})
    except (OSError, json.JSONDecodeError):
        return {}


def fetch(only: list[str] | None = None, on_progress=None) -> dict[str, int]:
    """Download the catalogue, skipping anything already present or missing."""
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
    have = installed()
    wanted = [p for p in CATALOGUE if not only or p.id in only]
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for photo in wanted:
            target = BACKGROUND_DIR / f"{photo.id}.jpg"
            if target.is_file() and photo.id in have:
                counts["skipped"] += 1
                continue
            try:
                with client.stream("GET", url_for(photo)) as response:
                    if response.status_code != 200:
                        counts["failed"] += 1
                        if on_progress:
                            on_progress(photo, f"unavailable ({response.status_code})")
                        continue
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_PHOTO_BYTES:
                            break
                if not body or len(body) > MAX_PHOTO_BYTES:
                    counts["failed"] += 1
                    if on_progress:
                        on_progress(photo, "empty" if not body
                                    else f"over {MAX_PHOTO_BYTES // 1_000_000}MB")
                    continue
                target.write_bytes(bytes(body))
            except httpx.HTTPError as exc:
                counts["failed"] += 1
                if on_progress:
                    on_progress(photo, f"failed: {type(exc).__name__}")
                continue

            have[photo.id] = {
                "name": photo.name, "category": photo.category,
                "file": target.name, "bytes": target.stat().st_size,
                "source": source_page(photo), "licence": LICENCE,
            }
            counts["downloaded"] += 1
            if on_progress:
                on_progress(photo, f"{target.stat().st_size // 1024}KB")

    MANIFEST.write_text(json.dumps(
        {"fetched_at": time.time(), "licence": LICENCE, "photos": have},
        indent=2), encoding="utf-8")
    return counts


def available() -> list[dict]:
    """What is on disk, for the dashboard picker."""
    have = installed()
    return [{"id": key, **value} for key, value in sorted(have.items())
            if (BACKGROUND_DIR / value["file"]).is_file()]
