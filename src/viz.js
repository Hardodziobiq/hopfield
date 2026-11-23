const canvas = document.getElementById('bitmap');
const context = canvas.getContext('2d');
const image = context.createImageData(canvas.width, canvas.height);
const slider = document.getElementById('frame');
const status = document.getElementById('status');
const back = document.getElementById('back');
const next = document.getElementById('next');
const play = document.getElementById('play');
const last = states.length - 1;
let timer = null;
let frame = 0;
slider.max = last;
slider.disabled = last === 0;
play.disabled = last === 0;

function show(index) {
  frame = index;
  const state = states[frame];
  for (let i = 0; i < state.length; i++) {
    const shade = state[i] === '1' ? 0 : 255;
    image.data.set([shade, shade, shade, 255], 4 * i);
  }
  context.putImageData(image, 0, 0);
  slider.value = frame;
  status.textContent = frame + ' / ' + last + (frame === last ? ' · ' + result : '');
  back.disabled = frame === 0;
  next.disabled = frame === last;
}

function stop() {
  clearInterval(timer);
  timer = null;
  play.textContent = 'Play';
}

function seek(index) {
  stop();
  show(index);
}

slider.addEventListener('input', () => seek(Number(slider.value)));
back.addEventListener('click', () => seek(frame - 1));
next.addEventListener('click', () => seek(frame + 1));
play.addEventListener('click', () => {
  if (timer !== null) {
    stop();
    return;
  }
  if (frame === last) show(0);
  play.textContent = 'Pause';
  timer = setInterval(() => {
    show(frame + 1);
    if (frame === last) stop();
  }, 550);
});

show(0);
