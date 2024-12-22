let resultDiv = document.getElementById("result") as HTMLDivElement;


function display_retrieval(data: string[]) {
    resultDiv.innerHTML = "";
    let len = data.length;
    let divWidth = resultDiv.clientWidth;

    data.forEach((imgSrc) => {
        let img = document.createElement("img");
        img.src = `data:image/png;base64,${imgSrc}`;
        img.width = divWidth / len;
        resultDiv.appendChild(img);
    });
}


