const statusEl = document.getElementById('status');
const vkEl = document.getElementById('vk');
const tkrEl = document.getElementById('t_kr');

const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/entrainment`);

ws.onopen = () => {
    statusEl.innerText = "Status: Metabolizing";
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    vkEl.innerText = `VK: B:${data.vk.bio.toFixed(2)} A:${data.vk.aff.toFixed(2)} S:${data.vk.soc.toFixed(2)} C:${data.vk.cog.toFixed(2)}`;
    tkrEl.innerText = `t_KR: ${data.t_kr}s`;
};

ws.onclose = () => {
    statusEl.innerText = "Status: Disconnected";
};
