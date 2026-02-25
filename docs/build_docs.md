# Build docs
## Prerequisites
The following prerequisites are required to build the documentation for resplotlib:

1. Install [Pandoc](https://github.com/jgm/pandoc/releases/tag/3.7.0.2) (for windows use .msi installer)

## Installation with dev dependencies
To install resplotlib with the necessary dependencies for building the documentation, follow these steps:

1. Clone the [resplotlib](https://github.com/Ruben-White/resplotlib) repository:
    ```bash
    git clone https://github.com/Ruben-White/resplotlib.git c:\...\resplotlib
    ```

2. Navigate to the cloned repository:
    ```bash
    cd c:\...\resplotlib
    ```

3. Synchronise the virtual environment
    ``` bash
    uv sync --extra docs
    ```

## Building the documentation
To build the documentation, follow these steps:
    
1. Build the documentation using [sphinx](https://www.sphinx-doc.org/en/master/):
    ```bash
    sphinx-build -b html docs/source docs/build/html
    ```