const canvas = document.getElementById('bitmap');
const slider = document.getElementById('frame');
const status = document.getElementById('status');
const back = document.getElementById('back');
const next = document.getElementById('next');
const play = document.getElementById('play');
const last = states.length - 1;
const delay = Math.max(60, Math.round(6000 / Math.max(last, 1)));
let timer = null;
let frame = 0;
slider.max = last;
slider.disabled = last === 0;
play.disabled = last === 0;

function draw(target, bits) {
  const context = target.getContext('2d');
  const image = context.createImageData(target.width, target.height);
  for (let i = 0; i < bits.length; i++) {
    const shade = bits[i] === '1' ? 0 : 255;
    image.data.set([shade, shade, shade, 255], 4 * i);
  }
  context.putImageData(image, 0, 0);
}

function show(index) {
  frame = index;
  draw(canvas, states[frame]);
  slider.value = frame;
  status.textContent = frame + ' / ' + last + ' · ' + labels[frame]
    + (frame === last ? ' · ' + result : '');
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
  }, delay);
});

draw(document.getElementById('original'), original);
show(0);