# Build documentation

## Prerequisites

The following prerequisites are required to build the documentation for this package:

1. Install [Pandoc](https://github.com/jgm/pandoc/releases/tag/3.7.0.2) (for windows use the .msi installer)

## Installation with docs dependencies

To install resplotlib with the necessary dependencies for building the documentation, follow these steps:

1. Clone the [resplotlib](https://github.com/Ruben-White/resplotlib) repository:

    ```bash
    git clone https://github.com/Ruben-White/resplotlib.git
    ```

2. Navigate to the root directory of the cloned repository:

    ```bash
    cd resplotlib
    ```

3. Synchronise the virtual environment with the extra dependencies for building the documentation:

    ```bash
    uv sync --extra docs
    ```

## Building the documentation

To build the documentation for this package, follow these steps:

1. Navigate to the root directory of the cloned repository:

    ```bash
    cd resplotlib
    ```

2. Build the documentation using [sphinx](https://www.sphinx-doc.org/en/master/):

    ```bash
    sphinx-build -b html docs/source docs/build/html
    ```
