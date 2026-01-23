const inputFile = document.querySelector("#file-input")
const imgArea = document.querySelector(".img-area")
const predictBtn = document.querySelector("#predict-btn")
const uploadBtn = document.querySelector(".btn-upload")

const triggerInput = () => inputFile.click()

if (uploadBtn) uploadBtn.addEventListener("click", triggerInput)
if (imgArea) imgArea.addEventListener("click", triggerInput)

if (inputFile) {
  inputFile.addEventListener("change", function () {
    const image = this.files[0]
    if (image) {
      if (image.size < 2000000) {
        const reader = new FileReader()
        reader.onload = () => {
          imgArea.innerHTML = ""

          const imgUrl = reader.result
          const img = document.createElement("img")
          img.src = imgUrl
          img.classList.add("analyzed-img")
          imgArea.appendChild(img)

          if (predictBtn) {
            predictBtn.style.display = "flex"
            predictBtn.classList.add("animate-fade-in")
          }
          if (uploadBtn) uploadBtn.style.display = "none"
        }
        reader.readAsDataURL(image)
      } else {
        alert("Image size must be less than 2MB")
        this.value = ""
      }
    }
  })
}

const speakBtn = document.querySelector("#speak-btn")
if (speakBtn) {
  speakBtn.addEventListener("click", () => {
    let textEn = speakBtn.getAttribute("data-en") || ""
    let textVi = speakBtn.getAttribute("data-vi") || ""

    if (!textEn.trim()) {
      const label = document.querySelector(".disease-label")?.innerText || "Result"
      textEn = `Diagnostic Result: ${label}.`
      textVi = `Kết quả chẩn đoán là ${label}.`
      console.warn("Speech data empty, using UI fallback:", { textEn })
    } else {
      console.log("Speech triggered with PDF/Server data:", { textEn, textVi })
    }

    if (!window.speechSynthesis) {
      console.error("Browser does not support Speech Synthesis")
      alert("Your browser does not support text-to-speech features.")
      return
    }

    window.speechSynthesis.cancel()

    const utteranceEn = new SpeechSynthesisUtterance(textEn)
    utteranceEn.lang = 'en-US'
    utteranceEn.rate = 0.9

    const utteranceVi = new SpeechSynthesisUtterance(textVi)
    utteranceVi.lang = 'vi-VN'
    utteranceVi.rate = 1.0
    speakBtn.classList.add("speaking")
    console.log("Speaking EN...")

    utteranceEn.onerror = (e) => console.error("EN Speech Error:", e)
    utteranceVi.onerror = (e) => console.error("VI Speech Error:", e)

    window.speechSynthesis.speak(utteranceEn)

    utteranceEn.onend = () => {
      console.log("Speaking VI...")
      window.speechSynthesis.speak(utteranceVi)
    }

    utteranceVi.onend = () => {
      console.log("Speech finished.")
      speakBtn.classList.remove("speaking")
    }

    setTimeout(() => {
      if (!window.speechSynthesis.speaking && speakBtn.classList.contains("speaking")) {
        speakBtn.classList.remove("speaking")
      }
    }, 10000)
  })
}
