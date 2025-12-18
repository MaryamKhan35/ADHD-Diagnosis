// EEG-ADHD Classification System - Main JavaScript
// Advanced neurodiagnostic webapp with interactive functionality

// Global variables
let uploadedFiles = [];
let analysisInProgress = false;
let brainwaveChart = null;
let currentFrequency = 'all';

// Mock EEG data for demonstration
const mockEEGData = {
    adhd_positive: {
        confidence: 87,
        classification: "ADHD Positive",
        thetaBetaRatio: 2.8,
        alphaPower: 45.2,
        processingSpeed: 156,
        signalQuality: 94,
        recommendations: [
            "Consider neurofeedback therapy sessions",
            "Implement structured daily routines",
            "Explore cognitive behavioral therapy",
            "Discuss medication options with healthcare provider"
        ]
    },
    adhd_negative: {
        confidence: 23,
        classification: "ADHD Negative",
        thetaBetaRatio: 1.2,
        alphaPower: 52.8,
        processingSpeed: 203,
        signalQuality: 97,
        recommendations: [
            "Continue current lifestyle practices",
            "Maintain regular sleep schedule",
            "Consider stress management techniques",
            "Monitor for any changes in attention patterns"
        ]
    }
};

// Brainwave data for visualization
const brainwaveData = {
    all: {
        delta: Array.from({length: 100}, (_, i) => Math.sin(i * 0.1) * 20 + Math.random() * 10),
        theta: Array.from({length: 100}, (_, i) => Math.sin(i * 0.15) * 30 + Math.random() * 15),
        alpha: Array.from({length: 100}, (_, i) => Math.sin(i * 0.2) * 25 + Math.random() * 12),
        beta: Array.from({length: 100}, (_, i) => Math.sin(i * 0.25) * 15 + Math.random() * 8),
        gamma: Array.from({length: 100}, (_, i) => Math.sin(i * 0.3) * 10 + Math.random() * 5)
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeNeuralBackground();
    initializeUploadZone();
    initializeScrollAnimations();
    initializeFrequencySelector();
    initializeBrainwaveChart();
});

// Neural network background animation
function initializeNeuralBackground() {
    const sketch = (p) => {
        let nodes = [];
        let connections = [];
        
        p.setup = () => {
            const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
            canvas.parent('neural-bg');
            
            // Create nodes
            for (let i = 0; i < 50; i++) {
                nodes.push({
                    x: p.random(p.width),
                    y: p.random(p.height),
                    vx: p.random(-0.5, 0.5),
                    vy: p.random(-0.5, 0.5),
                    size: p.random(2, 4)
                });
            }
        };
        
        p.draw = () => {
            p.clear();
            
            // Update and draw nodes
            nodes.forEach(node => {
                node.x += node.vx;
                node.y += node.vy;
                
                // Bounce off edges
                if (node.x < 0 || node.x > p.width) node.vx *= -1;
                if (node.y < 0 || node.y > p.height) node.vy *= -1;
                
                // Draw node
                p.fill(37, 99, 235, 100);
                p.noStroke();
                p.circle(node.x, node.y, node.size);
            });
            
            // Draw connections
            p.stroke(37, 99, 235, 30);
            p.strokeWeight(1);
            
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dist = p.dist(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
                    if (dist < 100) {
                        p.line(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
                    }
                }
            }
        };
        
        p.windowResized = () => {
            p.resizeCanvas(p.windowWidth, p.windowHeight);
        };
    };
    
    new p5(sketch);
}

// File upload functionality
function initializeUploadZone() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    // Click to upload
    uploadZone.addEventListener('click', () => {
        if (!analysisInProgress) {
            fileInput.click();
        }
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!analysisInProgress) {
            uploadZone.classList.add('dragover');
        }
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (!analysisInProgress) {
            handleFiles(e.dataTransfer.files);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', startAnalysis);
}

// Handle file uploads
function handleFiles(files) {
    const fileList = document.getElementById('file-list');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    // Clear previous files
    uploadedFiles = [];
    fileList.innerHTML = '';
    
    // Process new files
    Array.from(files).forEach(file => {
        if (validateFile(file)) {
            uploadedFiles.push(file);
            
            const fileItem = document.createElement('div');
            fileItem.className = 'file-info flex items-center justify-between';
            fileItem.innerHTML = `
                <div class="flex items-center">
                    <svg class="w-5 h-5 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span class="font-medium">${file.name}</span>
                </div>
                <div class="text-sm text-gray-500">
                    ${formatFileSize(file.size)}
                </div>
            `;
            fileList.appendChild(fileItem);
        }
    });
    
    // Enable analyze button if files are valid
    if (uploadedFiles.length > 0) {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
}

// Validate uploaded files
function validateFile(file) {
    const validExtensions = ['.edf', '.mat', '.csv'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    if (!validExtensions.includes(fileExtension)) {
        showNotification('Invalid file format. Please upload .edf, .mat, or .csv files.', 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('File size exceeds 100MB limit.', 'error');
        return false;
    }
    
    return true;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Start analysis process
function startAnalysis() {
    if (uploadedFiles.length === 0 || analysisInProgress) return;
    
    analysisInProgress = true;
    const analyzeBtn = document.getElementById('analyze-btn');
    const progressIndicator = document.getElementById('progress-indicator');
    
    // Disable analyze button
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
    analyzeBtn.textContent = 'Analyzing...';
    
    // Show progress indicator
    progressIndicator.classList.remove('hidden');
    
    // Simulate analysis steps
    simulateAnalysisSteps();
}

// Simulate analysis steps with realistic timing
function simulateAnalysisSteps() {
    const steps = [
        { id: 'step-1', duration: 1000 },
        { id: 'step-2', duration: 1500 },
        { id: 'step-3', duration: 2000 },
        { id: 'step-4', duration: 2500 },
        { id: 'step-5', duration: 1500 }
    ];
    
    let currentStep = 0;
    
    function processNextStep() {
        if (currentStep < steps.length) {
            const step = steps[currentStep];
            const stepElement = document.getElementById(step.id);
            
            // Mark step as active
            stepElement.classList.add('active');
            
            setTimeout(() => {
                // Mark step as completed
                stepElement.classList.remove('active');
                stepElement.classList.add('completed');
                stepElement.textContent = '✓';
                
                currentStep++;
                
                if (currentStep < steps.length) {
                    processNextStep();
                } else {
                    // Analysis complete
                    setTimeout(showResults, 500);
                }
            }, step.duration);
        }
    }
    
    processNextStep();
}

// Show analysis results
function showResults() {
    // Hide progress indicator
    document.getElementById('progress-indicator').classList.add('hidden');
    
    // Show visualization and results sections
    document.getElementById('visualization-section').classList.remove('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    
    // Generate random result (for demo purposes)
    const isAdhdPositive = Math.random() > 0.5;
    const resultData = isAdhdPositive ? mockEEGData.adhd_positive : mockEEGData.adhd_negative;
    
    // Update results
    updateClassificationResults(resultData);
    
    // Update brainwave chart
    updateBrainwaveChart();
    
    // Animate confidence meter
    animateConfidenceMeter(resultData.confidence);
    
    // Reset analysis state
    analysisInProgress = false;
    
    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

// Update classification results
function updateClassificationResults(data) {
    document.getElementById('theta-beta-ratio').textContent = data.thetaBetaRatio.toFixed(1);
    document.getElementById('alpha-power').textContent = data.alphaPower.toFixed(1) + ' μV²';
    document.getElementById('processing-speed').textContent = data.processingSpeed + ' ms';
    document.getElementById('signal-quality').textContent = data.signalQuality + '%';
    
    // Update classification result
    const resultElement = document.getElementById('classification-result');
    resultElement.textContent = data.classification;
    resultElement.className = data.confidence > 50 ? 'text-lg font-semibold text-red-600' : 'text-lg font-semibold text-green-600';
    
    // Update recommendations
    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = '';
    
    data.recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'flex items-start gap-2';
        item.innerHTML = `
            <svg class="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span class="text-sm text-gray-700">${rec}</span>
        `;
        recommendationsList.appendChild(item);
    });
}

// Animate confidence meter
function animateConfidenceMeter(targetPercentage) {
    const circle = document.getElementById('confidence-circle');
    const text = document.getElementById('confidence-text');
    const circumference = 2 * Math.PI * 50; // radius = 50
    
    let currentPercentage = 0;
    const increment = targetPercentage / 100;
    
    const animation = setInterval(() => {
        currentPercentage += increment;
        
        if (currentPercentage >= targetPercentage) {
            currentPercentage = targetPercentage;
            clearInterval(animation);
        }
        
        const offset = circumference - (currentPercentage / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        text.textContent = Math.round(currentPercentage) + '%';
    }, 20);
}

// Initialize brainwave chart
function initializeBrainwaveChart() {
    const chartContainer = document.getElementById('brainwave-chart');
    brainwaveChart = echarts.init(chartContainer);
    
    updateBrainwaveChart();
}

// Update brainwave chart
function updateBrainwaveChart() {
    if (!brainwaveChart) return;
    
    const data = brainwaveData.all;
    const timePoints = Array.from({length: 100}, (_, i) => i * 0.1);
    
    const option = {
        backgroundColor: 'transparent',
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: timePoints,
            axisLine: { lineStyle: { color: '#64748b' } },
            axisLabel: { color: '#64748b', fontSize: 10 }
        },
        yAxis: {
            type: 'value',
            axisLine: { lineStyle: { color: '#64748b' } },
            axisLabel: { color: '#64748b', fontSize: 10 },
            splitLine: { lineStyle: { color: '#e2e8f0' } }
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            borderColor: '#2563eb',
            textStyle: { color: '#374151' }
        },
        legend: {
            data: ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            textStyle: { color: '#64748b' },
            top: 10
        },
        series: [
            {
                name: 'Delta',
                type: 'line',
                data: data.delta,
                smooth: true,
                lineStyle: { color: '#dc2626', width: 2 },
                symbol: 'none'
            },
            {
                name: 'Theta',
                type: 'line',
                data: data.theta,
                smooth: true,
                lineStyle: { color: '#ea580c', width: 2 },
                symbol: 'none'
            },
            {
                name: 'Alpha',
                type: 'line',
                data: data.alpha,
                smooth: true,
                lineStyle: { color: '#2563eb', width: 2 },
                symbol: 'none'
            },
            {
                name: 'Beta',
                type: 'line',
                data: data.beta,
                smooth: true,
                lineStyle: { color: '#059669', width: 2 },
                symbol: 'none'
            },
            {
                name: 'Gamma',
                type: 'line',
                data: data.gamma,
                smooth: true,
                lineStyle: { color: '#7c3aed', width: 2 },
                symbol: 'none'
            }
        ]
    };
    
    brainwaveChart.setOption(option);
}

// Initialize frequency selector
function initializeFrequencySelector() {
    const frequencyButtons = document.querySelectorAll('.frequency-btn');
    
    frequencyButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            frequencyButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Update current frequency
            currentFrequency = button.dataset.frequency;
            
            // Update chart based on frequency
            updateChartByFrequency();
        });
    });
}

// Update chart based on selected frequency
function updateChartByFrequency() {
    if (!brainwaveChart) return;
    
    const option = brainwaveChart.getOption();
    
    if (currentFrequency === 'all') {
        // Show all series
        option.series.forEach(series => {
            series.show = true;
        });
    } else {
        // Show only selected frequency
        option.series.forEach(series => {
            series.show = series.name.toLowerCase() === currentFrequency;
        });
    }
    
    brainwaveChart.setOption(option);
}

// Scroll animations
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);
    
    // Observe all fade-in elements
    document.querySelectorAll('.fade-in').forEach(el => {
        observer.observe(el);
    });
}

// Utility functions
function scrollToUpload() {
    document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
}

function scrollToLearn() {
    document.getElementById('features-section').scrollIntoView({ behavior: 'smooth' });
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-20 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm ${
        type === 'error' ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Animate in
    anime({
        targets: notification,
        translateX: [300, 0],
        opacity: [0, 1],
        duration: 300,
        easing: 'easeOutQuad'
    });
    
    // Remove after 3 seconds
    setTimeout(() => {
        anime({
            targets: notification,
            translateX: [0, 300],
            opacity: [1, 0],
            duration: 300,
            easing: 'easeInQuad',
            complete: () => {
                document.body.removeChild(notification);
            }
        });
    }, 3000);
}

// Responsive chart resize
window.addEventListener('resize', () => {
    if (brainwaveChart) {
        brainwaveChart.resize();
    }
});

// Export functions for global access
window.scrollToUpload = scrollToUpload;
window.scrollToLearn = scrollToLearn;