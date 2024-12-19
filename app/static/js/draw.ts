let canvas = document.getElementById("draw") as HTMLCanvasElement;
let clearBtn = document.getElementById("clear") as HTMLButtonElement;
let submitBtn = document.getElementById("submit") as HTMLButtonElement;
let eraserCb = document.getElementById("eraser") as HTMLInputElement;
let ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

// 设置画布大小
let width = canvas.clientWidth;
let height = canvas.clientHeight;
canvas.width = width
canvas.height = height

// 添加偏移
let offsetX = canvas.getBoundingClientRect().left;
let offsetY = canvas.getBoundingClientRect().top;

// 绘图状态
let drawing = false;
let erasing = false;

// 上一个点的位置
let prevPos = { x: -1, y: -1 };

// 画笔
let lineWidth = (width + height) / 20;
let brushColor = "#000";
let backgroundColor = "#fff";
let styleColor = brushColor;

window.onresize = () => {
    console.log("窗口大小改变");
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = width
    canvas.height = height
    offsetX = canvas.getBoundingClientRect().left;
    offsetY = canvas.getBoundingClientRect().top;
    lineWidth = (width + height) / 20;
    reset()
}

// 鼠标按下事件
canvas.onmousedown = (e) => {
    drawing = true;
    let x = e.clientX - offsetX;
    let y = e.clientY - offsetY;
    prevPos = { x, y };
    console.log("鼠标按下", x, y);
    drawCircle(x, y, lineWidth / 2);
};

// 鼠标移动事件
canvas.onmousemove = (e) => {
    if (drawing) {
        let x = e.clientX - offsetX;
        let y = e.clientY - offsetY;
        let pos = { x, y };
        drawLine(prevPos.x, prevPos.y, pos.x, pos.y);
        console.log("鼠标移动", x, y);
        prevPos = pos;
    }
}

// 鼠标释放事件
canvas.onmouseup = () => {
    console.log("鼠标释放");
    drawing = false;
}

// 画圆
function drawCircle(x: number, y: number, r: number) {
    ctx.save();
    ctx.fillStyle = styleColor;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
}

// 画线
function drawLine(x1: number, y1: number, x2: number, y2: number) {
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = styleColor;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function reset() {
    console.log("重置画布", width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);
}
reset();

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