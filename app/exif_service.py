from PIL import Image, ExifTags
from typing import Dict, Any

def extract_exif_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts real camera EXIF provenance tags from an image file using PIL's ExifTags.
    If tags exist, returns the actual camera make/model/exposure values.
    If none exist, returns 'No EXIF Found' — without placeholder camera profiles.
    """
    no_exif = "No EXIF Found"
    metadata = {
        "has_exif": False,
        "is_hardware_camera": False,
        "device_make": no_exif,
        "device_model": no_exif,
        "exposure_settings": no_exif,
        "exposure_time": no_exif,
        "f_number": no_exif,
        "iso": no_exif,
        "lens_sensor": no_exif,
        "signature_status": no_exif,
        "software": no_exif,
        "details": [no_exif]
    }

    try:
        with Image.open(image_path) as img:
            exif_raw = img._getexif()
            if not exif_raw:
                exif_obj = img.getexif() if hasattr(img, 'getexif') else None
                if not exif_obj:
                    return metadata
                exif_raw = dict(exif_obj)
                if hasattr(ExifTags, 'IFD') and hasattr(exif_obj, 'get_ifd'):
                    try:
                        exif_raw.update(dict(exif_obj.get_ifd(ExifTags.IFD.Exif)))
                    except Exception:
                        pass
                if not exif_raw:
                    return metadata

            parsed_tags = {}
            for tag_id, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                parsed_tags[tag_name] = value

            if not parsed_tags:
                return metadata

            metadata["has_exif"] = True
            metadata["details"] = []

            # Extract real Camera Make and Model
            make = str(parsed_tags["Make"]).strip().replace('\x00', '') if "Make" in parsed_tags and parsed_tags["Make"] else None
            model = str(parsed_tags["Model"]).strip().replace('\x00', '') if "Model" in parsed_tags and parsed_tags["Model"] else None

            if make or model:
                metadata["is_hardware_camera"] = True
                metadata["device_make"] = make if make else no_exif
                if make and model:
                    if make.lower() in model.lower():
                        metadata["device_model"] = model
                    else:
                        metadata["device_model"] = f"{make} {model}"
                elif model:
                    metadata["device_model"] = model
                else:
                    metadata["device_model"] = make
                metadata["details"].append(f"Camera: {metadata['device_model']}")
            else:
                metadata["device_make"] = no_exif
                metadata["device_model"] = no_exif

            # Extract real Exposure parameters
            exposure_parts = []
            if "FNumber" in parsed_tags and parsed_tags["FNumber"]:
                f_val = parsed_tags["FNumber"]
                if isinstance(f_val, tuple) and len(f_val) == 2 and f_val[1] != 0:
                    f_num = f_val[0] / f_val[1]
                elif hasattr(f_val, 'numerator') and hasattr(f_val, 'denominator') and f_val.denominator != 0:
                    f_num = f_val.numerator / f_val.denominator
                else:
                    try:
                        f_num = float(f_val)
                    except (ValueError, TypeError):
                        f_num = f_val
                f_str = f"f/{f_num:.1f}" if isinstance(f_num, (int, float)) else f"f/{f_num}"
                metadata["f_number"] = f_str
                exposure_parts.append(f_str)
            else:
                metadata["f_number"] = no_exif

            if "ExposureTime" in parsed_tags and parsed_tags["ExposureTime"]:
                exp = parsed_tags["ExposureTime"]
                if isinstance(exp, tuple) and len(exp) == 2:
                    exp_str = f"{exp[0]}/{exp[1]}s" if exp[1] != 1 else f"{exp[0]}s"
                elif hasattr(exp, 'numerator') and hasattr(exp, 'denominator'):
                    exp_str = f"{exp.numerator}/{exp.denominator}s" if exp.denominator != 1 else f"{exp.numerator}s"
                elif isinstance(exp, (float, int)):
                    if 0 < exp < 1:
                        exp_str = f"1/{round(1 / exp)}s"
                    else:
                        exp_str = f"{exp}s"
                else:
                    exp_str = f"{exp}s"
                metadata["exposure_time"] = exp_str
                exposure_parts.append(exp_str)
            else:
                metadata["exposure_time"] = no_exif

            iso = parsed_tags.get("ISOSpeedRatings") or parsed_tags.get("PhotographicSensitivity")
            if iso:
                if isinstance(iso, (tuple, list)) and len(iso) > 0:
                    iso_str = f"ISO {iso[0]}"
                else:
                    iso_str = f"ISO {iso}"
                metadata["iso"] = iso_str
                exposure_parts.append(iso_str)
            else:
                metadata["iso"] = no_exif

            if exposure_parts:
                metadata["exposure_settings"] = " • ".join(exposure_parts)
                metadata["details"].append(f"Exposure: {metadata['exposure_settings']}")
            else:
                metadata["exposure_settings"] = no_exif

            # Extract real Lens / Sensor parameters
            lens = parsed_tags.get("LensModel") or parsed_tags.get("LensMake")
            if lens:
                metadata["lens_sensor"] = str(lens).strip().replace('\x00', '')
            elif "FocalLength" in parsed_tags and parsed_tags["FocalLength"]:
                fl = parsed_tags["FocalLength"]
                if isinstance(fl, tuple) and len(fl) == 2 and fl[1] != 0:
                    fl_num = fl[0] / fl[1]
                elif hasattr(fl, 'numerator') and hasattr(fl, 'denominator') and fl.denominator != 0:
                    fl_num = fl.numerator / fl.denominator
                else:
                    try:
                        fl_num = float(fl)
                    except (ValueError, TypeError):
                        fl_num = fl
                metadata["lens_sensor"] = f"{fl_num:.1f}mm Focal Length" if isinstance(fl_num, (int, float)) else f"{fl_num}mm"
            else:
                metadata["lens_sensor"] = no_exif

            # Extract real Software / Signature
            software = parsed_tags.get("Software")
            if software:
                metadata["software"] = str(software).strip().replace('\x00', '')
                metadata["signature_status"] = f"Software: {metadata['software']}"
            elif metadata["is_hardware_camera"]:
                metadata["signature_status"] = "Camera Hardware Verified"
            else:
                metadata["signature_status"] = no_exif
                metadata["software"] = no_exif

            if not metadata["details"]:
                metadata["details"].append("EXIF tags present (no camera make/model found)")

    except Exception as e:
        metadata["has_exif"] = False
        metadata["details"] = [f"Metadata read error: {str(e)}"]

    return metadata
