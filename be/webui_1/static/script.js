const [input, imgArea, predictBtn, uploadBtn] = ['#file-input', '.img-area', '#predict-btn', '.btn-upload'].map(s => document.querySelector(s))

const trigger = () => input?.click();
[uploadBtn, imgArea].forEach(btn => btn?.addEventListener('click', trigger))

input?.addEventListener('change', ({ target }) => {
  const file = target.files[0]
  if (!file) return

  if (file.size >= 2e6) {
    alert("Image size must be less than 2MB")
    target.value = ""
    return
  }

  imgArea.innerHTML = ""
  imgArea.appendChild(Object.assign(document.createElement('img'), {
    src: URL.createObjectURL(file),
    className: 'analyzed-img'
  }))

  if (predictBtn) {
    predictBtn.style.display = 'flex'
    predictBtn.classList.add('animate-fade-in')
  }
  if (uploadBtn) uploadBtn.style.display = 'none'
})
