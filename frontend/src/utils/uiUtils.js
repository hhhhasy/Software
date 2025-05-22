export function showLoading(message = '处理中...') {
    let loadingMask = document.getElementById('loadingMask');
    if (!loadingMask) {
        loadingMask = document.createElement('div');
        loadingMask.id = 'loadingMask';
        loadingMask.className = 'loading-mask'; // 样式在 base.css 中定义
        loadingMask.innerHTML = `<div class="loading-spinner"></div><p>${message}</p>`;
        document.body.appendChild(loadingMask);
    }
    loadingMask.style.display = 'flex';
     if (message) {
        const p = loadingMask.querySelector('p');
        if (p) p.textContent = message;
    }
}

export function hideLoading() {
    const loadingMask = document.getElementById('loadingMask');
    if (loadingMask) {
        loadingMask.style.display = 'none';
    }
}

export function showError(message, duration = 3000) {
    let errorContainer = document.getElementById('errorContainer');
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.id = 'errorContainer';
        errorContainer.className = 'error-message'; // 样式在 base.css 中定义
        document.body.appendChild(errorContainer);
    }
    errorContainer.textContent = message;
    errorContainer.style.display = 'block';

    if (errorContainer.timeoutId) {
        clearTimeout(errorContainer.timeoutId);
    }

    errorContainer.timeoutId = setTimeout(() => {
        errorContainer.style.display = 'none';
    }, duration);
}

export function showSuccess(message, duration = 3000) {
    let successContainer = document.getElementById('successContainer');
    if (!successContainer) {
        successContainer = document.createElement('div');
        successContainer.id = 'successContainer';
        successContainer.className = 'success-message'; // 样式在 base.css 中定义
        document.body.appendChild(successContainer);
    }
    successContainer.textContent = message;
    successContainer.style.display = 'block';

    if (successContainer.timeoutId) {
        clearTimeout(successContainer.timeoutId);
    }

    successContainer.timeoutId = setTimeout(() => {
        successContainer.style.display = 'none';
    }, duration);
}
