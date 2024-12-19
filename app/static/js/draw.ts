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