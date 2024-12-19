let canvas = document.getElementById("draw") as HTMLCanvasElement;
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

// 调整画布大小
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

// 重置画布
function reset() {
    console.log("重置画布", width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);
}
reset();
