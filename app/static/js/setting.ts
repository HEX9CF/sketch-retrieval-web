let clearBtn = document.getElementById("clear") as HTMLButtonElement;
let eraserCb = document.getElementById("eraser") as HTMLInputElement;
let brushColorInput = document.getElementById("brush-color") as HTMLInputElement;
let backgroundColorInput = document.getElementById("background-color") as HTMLInputElement;
let brushSizeInput = document.getElementById("brush-size") as HTMLInputElement;

// 清空画布
clearBtn.onclick = () => {
    console.log("清空画布",  width, height);
    reset();
}

// 橡皮擦
eraserCb.onchange = () => {
    console.log("橡皮擦", eraserCb.checked);
    erasing = eraserCb.checked;
    styleColor = erasing ? backgroundColor : brushColor;
}

// 笔刷颜色
brushColorInput.onchange = () => {
    console.log("笔刷颜色", brushColorInput.value);
    brushColor = brushColorInput.value;
    styleColor = erasing ? backgroundColor : brushColor;
}

// 背景颜色
backgroundColorInput.onchange = () => {
    console.log("背景颜色", backgroundColorInput.value);
    backgroundColor = backgroundColorInput.value;
}

// 笔刷大小
brushSizeInput.onchange = () => {
    let percent = parseInt(brushSizeInput.value);
    lineWidth = (width + height) * percent / 1000;
    if (lineWidth < 1) {
        lineWidth = 1;
    }
    console.log("笔刷大小", brushSizeInput.value, lineWidth);
}
