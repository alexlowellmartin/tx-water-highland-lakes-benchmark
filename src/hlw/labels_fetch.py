"""Entry point: download static weak-label layers (WorldCover, IO LULC) onto lake grids."""

from .labels import fetch_static_labels


def main():
    fetch_static_labels()


if __name__ == "__main__":
    main()
