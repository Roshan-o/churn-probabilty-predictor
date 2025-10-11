// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const fileSelected = document.getElementById('fileSelected');
const uploadBtn = document.getElementById('uploadBtn');
const uploadForm = document.getElementById('uploadForm');

// File upload functionality
if (fileInput && uploadArea) {
    // File input change event
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            showFileSelected(file);
        }
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isValidFile(file)) {
                fileInput.files = files;
                showFileSelected(file);
            } else {
                showAlert('Please select a valid CSV or Excel file.', 'warning');
            }
        }
    });

    // Form submission with loading state
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (!fileInput.files[0]) {
                e.preventDefault();
                showAlert('Please select a file first.', 'warning');
                return;
            }
            
            // Show loading state
            uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
            uploadBtn.disabled = true;
            
            // Add loading class to upload area
            uploadArea.classList.add('loading');
        });
    }
}

// Show file selected state
function showFileSelected(file) {
    if (uploadPlaceholder && fileSelected) {
        uploadPlaceholder.classList.add('d-none');
        fileSelected.classList.remove('d-none');
        
        const fileName = fileSelected.querySelector('.file-name');
        if (fileName) {
            fileName.textContent = file.name;
        }
        
        if (uploadBtn) {
            uploadBtn.disabled = false;
        }
    }
}

// Clear selected file
function clearFile() {
    if (fileInput) {
        fileInput.value = '';
    }
    
    if (uploadPlaceholder && fileSelected) {
        uploadPlaceholder.classList.remove('d-none');
        fileSelected.classList.add('d-none');
    }
    
    if (uploadBtn) {
        uploadBtn.disabled = true;
    }
    
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
}

// Validate file type
function isValidFile(file) {
    const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    const validExtensions = ['.csv', '.xls', '.xlsx'];
    
    const hasValidType = validTypes.includes(file.type);
    const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    
    return hasValidType || hasValidExtension;
}

// Show alert message
function showAlert(message, type = 'info') {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        <i class="fas fa-info-circle me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    if (container && container.firstChild) {
        container.insertBefore(alert, container.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert && alert.parentNode) {
            alert.classList.remove('show');
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 150);
        }
    }, 5000);
}

// File size formatter
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Animate elements on page load
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to main elements
    const elementsToAnimate = document.querySelectorAll('.upload-card, .feature-item, .card');
    elementsToAnimate.forEach((element, index) => {
        setTimeout(() => {
            element.classList.add('fade-in-up');
        }, index * 100);
    });
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add loading animation to buttons
function addLoadingToButton(button, text = 'Loading...') {
    if (button) {
        const originalText = button.innerHTML;
        button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${text}`;
        button.disabled = true;
        
        return function() {
            button.innerHTML = originalText;
            button.disabled = false;
        };
    }
}

// Utility function to show/hide elements
function toggleElement(element, show = true) {
    if (element) {
        if (show) {
            element.classList.remove('d-none');
        } else {
            element.classList.add('d-none');
        }
    }
}

// Progress bar animation (for future use)
function animateProgressBar(progressBar, targetValue, duration = 1000) {
    if (!progressBar) return;
    
    let startValue = 0;
    const startTime = Date.now();
    
    function update() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = startValue + (targetValue - startValue) * progress;
        progressBar.style.width = currentValue + '%';
        progressBar.setAttribute('aria-valuenow', currentValue);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// Export functions for global use
window.clearFile = clearFile;
window.showAlert = showAlert;
window.formatFileSize = formatFileSize;
window.addLoadingToButton = addLoadingToButton;
window.toggleElement = toggleElement;
window.animateProgressBar = animateProgressBar;