let submitBtn = document.getElementById("submit") as HTMLButtonElement;

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
