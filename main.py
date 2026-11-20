import argparse
import csv
import math
from pathlib import Path

from PIL import Image
import numpy as np

from noise3d import get_3dnoise, make_tiff_data_cube, set_3dnoise


NOISE_KEYS = (
	"sigma_v",
	"sigma_h",
	"sigma_t",
	"sigma_th",
	"sigma_tv",
	"sigma_vh",
	"sigma_tvh",
)
TIFF_SUFFIXES = {".tif", ".tiff"}


def has_tiffs(directory):
	return any(path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
			   for path in directory.iterdir())


def find_sequences(sequence_root):
	if not sequence_root.is_dir():
		raise ValueError(f"Sequence folder does not exist: {sequence_root}")

	if has_tiffs(sequence_root):
		return [sequence_root]

	sequences = sorted(
		path for path in sequence_root.iterdir()
		if path.is_dir() and has_tiffs(path)
	)
	if not sequences:
		raise ValueError(
			"No TIFF sequences found. Put each sequence in its own folder, "
			"or put one sequence directly in the supplied folder."
		)
	return sequences


def measure_sequence(sequence_path):
	data_cube = make_tiff_data_cube(str(sequence_path))
	if data_cube.shape[0] < 2:
		raise ValueError(f"A sequence needs at least two frames: {sequence_path}")

	components = get_3dnoise(data_cube)
	magnitudes = {
		key: float(np.mean(np.asarray(components[key], dtype=float)))
		for key in NOISE_KEYS
	}
	noise_rms = math.sqrt(sum(value ** 2 for value in magnitudes.values()))
	mean_signal = float(np.mean(components["signal"]))
	relative_noise = (
		noise_rms / abs(mean_signal) * 100
		if not np.isclose(mean_signal, 0)
		else float("nan")
	)
	score = relative_noise if np.isfinite(relative_noise) else noise_rms

	return {
		"name": sequence_path.name,
		"mean_signal": mean_signal,
		"noise_rms": noise_rms,
		"relative_noise_percent": relative_noise,
		"score": score,
		"tier": "",
		"components": components,
	}


def classify_sequences(records):
	scores = np.asarray([record["score"] for record in records])
	low_cutoff, high_cutoff = np.quantile(scores, [1 / 3, 2 / 3])
	for record in records:
		if record["score"] < low_cutoff:
			record["tier"] = "low"
		elif record["score"] > high_cutoff:
			record["tier"] = "high"
		else:
			record["tier"] = "medium"


def load_measurements(sequence_root):
	records = [measure_sequence(path) for path in find_sequences(sequence_root)]
	classify_sequences(records)
	return records


def print_report(records):
	print("Sequence                 Mean signal    Noise RMS    Relative noise    Tier")
	for record in records:
		relative = record["relative_noise_percent"]
		relative_text = f"{relative:.3f}%" if np.isfinite(relative) else "n/a"
		print(
			f"{record['name'][:24]:24} "
			f"{record['mean_signal']:12.3f} "
			f"{record['noise_rms']:11.3f} "
			f"{relative_text:17} "
			f"{record['tier']}"
		)


def save_report(records, report_path):
	fields = (
		"sequence",
		"mean_signal",
		"noise_rms",
		"relative_noise_percent",
		"tier",
	)
	with report_path.open("w", newline="", encoding="utf-8") as report_file:
		writer = csv.DictWriter(report_file, fieldnames=fields)
		writer.writeheader()
		for record in records:
			writer.writerow({
				"sequence": record["name"],
				"mean_signal": record["mean_signal"],
				"noise_rms": record["noise_rms"],
				"relative_noise_percent": record["relative_noise_percent"],
				"tier": record["tier"],
			})


def apply_noise(records, level):
	candidates = [record for record in records if record["tier"] == level]
	if not candidates:
		raise ValueError(
			f"No sequence was classified as {level}. "
			"Use more sequences with a wider range of measured noise."
		)

	median_score = float(np.median([record["score"] for record in candidates]))
	return min(candidates, key=lambda record: abs(record["score"] - median_score))


def apply_command(args):
	records = load_measurements(args.sequences)
	profile = apply_noise(records, args.level)

	with Image.open(args.clean_image) as image:
		clean_image = np.array(image)
	if clean_image.ndim != 2:
		raise ValueError("The clean image must be a grayscale TIFF")
	if args.clean_image.resolve() == args.output.resolve():
		raise ValueError("Output path must not overwrite the clean image")

	components = {
		key: profile["components"][key]
		for key in NOISE_KEYS
	}
	components["signal"] = clean_image[np.newaxis, :, :]
	noisy_image = set_3dnoise(
		components, 1, clean_image.shape[0], clean_image.shape[1]
	)[0]

	if np.issubdtype(clean_image.dtype, np.integer):
		limits = np.iinfo(clean_image.dtype)
		noisy_image = np.clip(np.rint(noisy_image), limits.min, limits.max)
	noisy_image = noisy_image.astype(clean_image.dtype)
	Image.fromarray(noisy_image).save(args.output)
	print(
		f"Applied {args.level} noise from sequence {profile['name']!r}; "
		f"saved {args.output}"
	)


def main():
	parser = argparse.ArgumentParser(
		description="Measure camera sequence noise and apply a measured noise tier."
	)
	commands = parser.add_subparsers(dest="command", required=True)

	analyze_parser = commands.add_parser(
		"analyze", help="compare and classify TIFF sequences" 
	)
	analyze_parser.add_argument(
		"sequences", type=Path,
		help="folder containing one subfolder per sequence, or one sequence",
	)
	analyze_parser.add_argument(
		"--report", type=Path, help="optional path for a CSV noise report"
	)

	apply_parser = commands.add_parser(
		"apply", help="apply a measured noise tier to a clean grayscale TIFF"
	)
	apply_parser.add_argument("sequences", type=Path)
	apply_parser.add_argument("clean_image", type=Path)
	apply_parser.add_argument("--level", choices=("low", "medium", "high"), required=True)
	apply_parser.add_argument("--output", type=Path, required=True)

	args = parser.parse_args()
	try:
		if args.command == "analyze":
			records = load_measurements(args.sequences)
			print_report(records)
			if args.report:
				save_report(records, args.report)
				print(f"CSV report saved to {args.report}")
		else:
			apply_command(args)
	except (OSError, TypeError, ValueError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()