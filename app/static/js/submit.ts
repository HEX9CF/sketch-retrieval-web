declare var bootstrap: any;
let submitBtn = document.getElementById("submit") as HTMLButtonElement;
let toastElem = document.getElementById("toast") as HTMLDivElement;
let toastBody = document.getElementById("toast-body") as HTMLDivElement;
let toastTitle = document.getElementById("toast-title") as HTMLDivElement;
let toast = new bootstrap.Toast(toastElem);

// 提交画布
submitBtn.onclick = () => {
    console.log("提交画布");
    canvas.toBlob((blob) => {
        if (blob) {
            let formData = new FormData();
            formData.append("file", blob, "draw.png");
            console.log(blob)
            fetch("/api/recognize", {
                method: "POST",
                body: formData
            }).then(res => res.json())
                .then((data) => {
                    console.log(data);
                    if (data["code"] === 1) {
                        // alert("识别结果：" + data["data"]);
                        showToast("识别结果", '<h1>' + data["data"] + '</h1>');
                    } else {
                        // alert("识别失败：" + data["msg"]);
                        showToast("识别失败", data["msg"]);
                    }
                }).catch((err) => {
                // alert("识别失败：" + err);
                showToast("识别失败", err);
            });
        }
    }, "image/png");
}

function showToast(title: string, msg: string) {
    console.log("Toast", title, msg);
    toastTitle.innerHTML = title;
    toastBody.innerHTML = msg;
    toast.show();
}