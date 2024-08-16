import marimo

__generated_with = "0.7.20"
app = marimo.App(width="medium")


@app.cell
def __():
    from typing import get_args

    import marimo as mo
    from dotenv import load_dotenv

    from hsgp_lvm.enum import HSGP_LVM, MODEL_NAME
    from hsgp_lvm.results import ModelResult

    _ = load_dotenv(override=True)
    return HSGP_LVM, MODEL_NAME, ModelResult, get_args, load_dotenv, mo


@app.cell
def __(HSGP_LVM, MODEL_NAME, get_args, mo):
    model_name = mo.ui.dropdown(
        options=list(get_args(MODEL_NAME)), label="Select Model:", value=HSGP_LVM
    )
    model_name
    return (model_name,)


@app.cell
def __(ModelResult, model_name):
    result = ModelResult.load(model_name.value)
    return (result,)


@app.cell
def __(result):
    result
    return


if __name__ == "__main__":
    app.run()
