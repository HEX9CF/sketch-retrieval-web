let canvas = document.getElementById("draw") as HTMLCanvasElement;
let clearBtn = document.getElementById("clear") as HTMLButtonElement;
let submitBtn = document.getElementById("submit") as HTMLButtonElement;
let ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

let width = canvas.clientWidth;
let height = canvas.clientHeight;

canvas.width = width
canvas.height = height

// 添加偏移
let offsetX = canvas.getBoundingClientRect().left;
let offsetY = canvas.getBoundingClientRect().top;

let drawing = false;
let erasing = false;

let prevPos = { x: -1, y: -1 };

// 鼠标按下事件
canvas.onmousedown = (e) => {
    drawing = true;
    let x = e.clientX - offsetX;
    let y = e.clientY - offsetY;
    prevPos = { x, y };
    console.log("鼠标按下", x, y);
    drawCircle(x, y, 5);
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
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
}

// 画线
function drawLine(x1: number, y1: number, x2: number, y2: number) {
    ctx.save();
    ctx.beginPath();
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.closePath();
}

clearBtn.onclick = () => {
    console.log("清空画布", offsetX, offsetY, offsetX + width, offsetY + height);
    ctx.clearRect(offsetX, offsetY, offsetX + width, offsetY + height);
}

submitBtn.onclick = () => {
    console.log("提交画布");
    canvas.toBlob((blob) => {
        if (blob) {
            let formData = new FormData();
            formData.append("file", blob, "draw.png");
            fetch("/recognize", {
                method: "POST",
                body: formData
            }).then(res => res.json())
                .then((data) => {
                    console.log(data);
                    if (data["code"] === 1) {
                        alert("识别结果：" + data["data"]);
                    } else {
                        alert("识别失败：" + data["msg"]);
                    }
                }).catch((err) => {
                    alert("识别失败：" + err);
            });
        }
    }, "image/png");
}