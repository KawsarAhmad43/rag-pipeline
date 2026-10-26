document.addEventListener('DOMContentLoaded', () => {
  const status = document.getElementById('status');
  const reload = document.getElementById('reload');
  const clear = document.getElementById('clear-cache');

  reload.addEventListener('click', async () => {
    status.textContent = 'Status: reloading...';
    try {
      // Placeholder for backend call
      await new Promise(r => setTimeout(r, 600));
      status.textContent = 'Status: reloaded (demo)';
    } catch (e) {
      status.textContent = 'Status: error';
    }
  });

  clear.addEventListener('click', async () => {
    status.textContent = 'Status: clearing cache...';
    await new Promise(r => setTimeout(r, 400));
    status.textContent = 'Status: cache cleared (demo)';
  });
});
