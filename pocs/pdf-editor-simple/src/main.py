import argparse
import os
import sys

import editor
from sample import write_sample


def main():
    parser = argparse.ArgumentParser(prog="pdf-editor", description="Simple PDF editor")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="page count, page sizes and rotation")
    info.add_argument("pdf")

    merge = commands.add_parser("merge", help="join several PDFs into one")
    merge.add_argument("pdf", nargs="+")
    merge.add_argument("-o", "--out", required=True)

    split = commands.add_parser("split", help="write every page as its own file")
    split.add_argument("pdf")
    split.add_argument("-o", "--out", required=True, help="output directory")

    extract = commands.add_parser("extract", help="keep the given pages, in the given order")
    extract.add_argument("pdf")
    extract.add_argument("-p", "--pages", required=True, help="such as 1-3,7")
    extract.add_argument("-o", "--out", required=True)

    delete = commands.add_parser("delete", help="drop the given pages")
    delete.add_argument("pdf")
    delete.add_argument("-p", "--pages", required=True)
    delete.add_argument("-o", "--out", required=True)

    rotate = commands.add_parser("rotate", help="turn the given pages clockwise")
    rotate.add_argument("pdf")
    rotate.add_argument("-p", "--pages", required=True)
    rotate.add_argument("-a", "--angle", type=int, default=90)
    rotate.add_argument("-o", "--out", required=True)

    text = commands.add_parser("text", help="print the text of the given pages")
    text.add_argument("pdf")
    text.add_argument("-p", "--pages", default="")

    sample = commands.add_parser("sample", help="write a small multi page PDF to work on")
    sample.add_argument("-o", "--out", required=True)
    sample.add_argument("-n", "--pages", type=int, default=4)

    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def run(args):
    if args.command == "info":
        report = editor.info(args.pdf)
        print(f"{args.pdf}: {len(report['pages'])} pages, encrypted={report['encrypted']}")
        if report["title"]:
            print(f"title: {report['title']}")
        if report["producer"]:
            print(f"producer: {report['producer']}")
        for number, width, height, rotation in report["pages"]:
            print(f"page {number}: {width}x{height} pt, rotation {rotation}")

    elif args.command == "merge":
        print(editor.merge(args.pdf, args.out))

    elif args.command == "split":
        os.makedirs(args.out, exist_ok=True)
        for path in editor.split(args.pdf, args.out):
            print(path)

    elif args.command == "extract":
        print(editor.extract(args.pdf, args.pages, args.out))

    elif args.command == "delete":
        print(editor.delete(args.pdf, args.pages, args.out))

    elif args.command == "rotate":
        print(editor.rotate(args.pdf, args.pages, args.angle, args.out))

    elif args.command == "text":
        for number, body in editor.text(args.pdf, args.pages):
            print(f"--- page {number}")
            print(body)

    elif args.command == "sample":
        titles = [f"Page {number}" for number in range(1, args.pages + 1)]
        print(write_sample(args.out, titles))


if __name__ == "__main__":
    sys.exit(main())
