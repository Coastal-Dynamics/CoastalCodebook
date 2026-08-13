# Building the book

If you'd like to build the book yourself locally, you can do so using Quarto. Quarto needs to be installed separately from the other CoastalCodebook software requirements. Here we assume you have already installed these requirements as per our [installation instructions](installation.md), including registering the `coastalcodebook` environment as a Jupyter kernel.

## Install Quarto

Follow the official Quarto installation instructions for your operating system.

For Ubuntu/Debian Linux, download the `.deb` package from the Quarto download page and install it with:

```bash
sudo dpkg -i quarto-<version>-linux-amd64.deb
```

You can check that Quarto is installed correctly with: `quarto --version`.

## Build the book

1. Navigate to the Quarto directory: `cd ~/path/to/CoastalCodebook/quarto`
2. Activate the environment by running `mamba activate coastalcodebook`
3. Build the book: `quarto render --execute`

A fully rendered HTML version of the book will be built in:

`~/path/to/CoastalCodebook/quarto/_site/`

You can view the generated files in `_site/` using your preferred method. For instance, you can serve the `_site` directory locally:

```bash
python -m http.server 8000 --directory _site
```

And open http://localhost:8000/ in your browser.