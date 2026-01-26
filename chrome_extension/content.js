// Content script - runs on all web pages
// Allows highlighting text and detecting stage

console.log('SAP Stage Detector loaded');

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getSelectedText') {
        sendResponse({ text: window.getSelection().toString() });
    }
});
