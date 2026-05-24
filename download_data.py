import argparse
import os

from quickdraw import QuickDrawDataGroup


DEFAULT_CLASSES = ["ant", "cat", "dog"]
DEFAULT_CACHE_DIR = ".quickdrawcache"
DEFAULT_IMAGE_DIR = "data/quickdraw"


def make_dirs(*dir_names):
    for dir_name in dir_names:
        os.makedirs(dir_name, exist_ok=True)


def generate_class_images(
    class_names,
    cache_dir,
    image_dir,
    stroke_widths,
    max_drawings=1000,
    drawing_size=(28, 28),
    recognized=True,
):
    if not class_names:
        raise ValueError("Provide at least one class name.")

    for class_name in class_names:
        drawing_group = QuickDrawDataGroup(
            class_name,
            max_drawings=max_drawings,
            recognized=recognized,
            cache_dir=cache_dir,
        )

        for index, drawing in enumerate(drawing_group.drawings):
            for width in stroke_widths:
                class_dir = os.path.join(image_dir, class_name)
                make_dirs(class_dir)
                image_path = os.path.join(
                    class_dir,
                    f"{class_name}_{index}_width_{width}.png",
                )
                drawing.get_image(stroke_width=width).resize(drawing_size).save(image_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Download Google QuickDraw images.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--max-drawings", type=int, default=1000)
    parser.add_argument("--stroke-widths", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--include-unrecognized", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    make_dirs(args.cache_dir, args.image_dir)
    generate_class_images(
        class_names=args.classes,
        cache_dir=args.cache_dir,
        image_dir=args.image_dir,
        stroke_widths=args.stroke_widths,
        max_drawings=args.max_drawings,
        drawing_size=(args.size, args.size),
        recognized=None if args.include_unrecognized else True,
    )


if __name__ == "__main__":
    main()
