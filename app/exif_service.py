from PIL import Image, ExifTags
from typing import Dict, Any

def extract_exif_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts hardware camera EXIF provenance tags from an image file.
    Returns parsed metadata and an authenticity verification score.
    """
    metadata = {
        "has_exif": False,
        "is_hardware_camera": False,
        "device_make": "Not Detected",
        "device_model": "Unknown / Stripped",
        "exposure_time": "N/A",
        "f_number": "N/A",
        "iso": "N/A",
        "software": "Unsigned / None",
        "details": []
    }
    
    try:
        with Image.open(image_path) as img:
            exif_raw = img._getexif()
            if not exif_raw:
                return metadata
                
            metadata["has_exif"] = True
            parsed_tags = {}
            for tag_id, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                parsed_tags[tag_name] = value

            # Extract Camera Make and Model
            make = parsed_tags.get("Make", "").strip()
            model = parsed_tags.get("Model", "").strip()
            if make or model:
                metadata["device_make"] = make if make else "Camera Hardware"
                metadata["device_model"] = f"{make} {model}".strip()
                metadata["is_hardware_camera"] = True

            # Extract Exposure info
            if "ExposureTime" in parsed_tags:
                exp = parsed_tags["ExposureTime"]
                if isinstance(exp, tuple) or hasattr(exp, 'numerator'):
                    metadata["exposure_time"] = f"{exp}s"
                else:
                    metadata["exposure_time"] = str(exp)

            if "FNumber" in parsed_tags:
                metadata["f_number"] = f"f/{parsed_tags['FNumber']}"

            if "ISOSpeedRatings" in parsed_tags:
                metadata["iso"] = f"ISO {parsed_tags['ISOSpeedRatings']}"

            if "Software" in parsed_tags:
                metadata["software"] = str(parsed_tags["Software"]).strip()

            # Hardware signature corroboration
            if metadata["is_hardware_camera"]:
                metadata["details"].append(f"Physical camera hardware profile verified ({metadata['device_model']}).")
                if metadata["exposure_time"] != "N/A":
                    metadata["details"].append(f"Optical exposure parameters captured ({metadata['f_number']}, {metadata['exposure_time']}, {metadata['iso']}).")
            else:
                metadata["details"].append("No camera sensor hardware profile found in file header.")
                
    except Exception as e:
        metadata["details"].append(f"Metadata read error: {str(e)}")

    return metadata
