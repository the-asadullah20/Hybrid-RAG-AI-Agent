// Theme Management
let currentTheme = localStorage.getItem('theme') || 'light';
let currentColor = localStorage.getItem('themeColor') || 'purple';

const themeColors = {
    blue: { light: '#3B82F6', dark: '#2563EB', lightBg: '#dbeafe' },
    purple: { light: '#8B5CF6', dark: '#7c3aed', lightBg: '#ede9fe' },
    green: { light: '#10B981', dark: '#059669', lightBg: '#d1fae5' },
    orange: { light: '#F59E0B', dark: '#d97706', lightBg: '#fef3c7' },
    pink: { light: '#EC4899', dark: '#db2777', lightBg: '#fce7f3' },
    red: { light: '#EF4444', dark: '#dc2626', lightBg: '#fee2e2' }
};

// Initialize theme
function initTheme() {
    document.body.className = currentTheme === 'dark' ? 'dark-theme' : 'light-theme';
    const themeIcon = document.getElementById('themeIcon');
    if (themeIcon) {
        themeIcon.className = currentTheme === 'dark' 
            ? 'fas fa-sun' 
            : 'fas fa-moon';
    }
    applyThemeColor();
}

function applyThemeColor() {
    const color = themeColors[currentColor];
    document.documentElement.style.setProperty('--theme-color', color.light);
    document.documentElement.style.setProperty('--theme-color-dark', color.dark);
    document.documentElement.style.setProperty('--theme-color-light', color.lightBg);
    
    // Update active color button
    document.querySelectorAll('.color-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.color === currentColor) {
            btn.classList.add('active');
        }
    });
}

// Settings toggle
const settingsToggle = document.getElementById('settingsToggle');
const settingsPanel = document.getElementById('settingsPanel');
const settingsChevron = document.getElementById('settingsChevron');

if (settingsToggle && settingsPanel) {
    settingsToggle.addEventListener('click', () => {
        settingsPanel.classList.toggle('hidden');
        if (settingsChevron) {
            settingsChevron.style.transform = settingsPanel.classList.contains('hidden') 
                ? 'rotate(0deg)' 
                : 'rotate(180deg)';
        }
    });
}

// Theme toggle (moon/sun icon at top)
const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        const icon = document.getElementById('themeIcon');
        if (icon) {
            // Add rotation animation
            icon.style.transform = 'rotate(360deg) scale(0)';
            setTimeout(() => {
                currentTheme = currentTheme === 'light' ? 'dark' : 'light';
                localStorage.setItem('theme', currentTheme);
                initTheme();
                icon.style.transform = 'rotate(0deg) scale(1)';
            }, 200);
        } else {
            currentTheme = currentTheme === 'light' ? 'dark' : 'light';
            localStorage.setItem('theme', currentTheme);
            initTheme();
        }
    });
}

// Color picker
document.querySelectorAll('.color-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        currentColor = btn.dataset.color;
        localStorage.setItem('themeColor', currentColor);
        applyThemeColor();
    });
});

// View toggle (Chats/Statistics)
let currentView = 'chats';

document.getElementById('chatsViewBtn').addEventListener('click', () => {
    switchView('chats');
});

document.getElementById('statsViewBtn').addEventListener('click', () => {
    switchView('statistics');
});

function switchView(view) {
    currentView = view;
    
    // Add fade animation
    const chatsView = document.getElementById('chatsView');
    const statsView = document.getElementById('statisticsView');
    
    if (view === 'chats') {
        if (!statsView.classList.contains('hidden')) {
            statsView.style.opacity = '0';
            setTimeout(() => {
                statsView.classList.add('hidden');
                chatsView.classList.remove('hidden');
                chatsView.style.opacity = '0';
                setTimeout(() => {
                    chatsView.style.opacity = '1';
                }, 10);
            }, 150);
        } else {
            chatsView.classList.remove('hidden');
        }
    } else {
        if (!chatsView.classList.contains('hidden')) {
            chatsView.style.opacity = '0';
            setTimeout(() => {
                chatsView.classList.add('hidden');
                statsView.classList.remove('hidden');
                statsView.style.opacity = '0';
                setTimeout(() => {
                    statsView.style.opacity = '1';
                }, 10);
            }, 150);
        } else {
            statsView.classList.remove('hidden');
        }
    }
    
    // Update buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });
    
    // Show/hide new chat button
    document.getElementById('newChatBtn').style.display = view === 'chats' ? 'flex' : 'none';
}

// Chat Management
let currentChatId = null;

document.getElementById('newChatBtn').addEventListener('click', createNewChat);

async function createNewChat() {
    try {
        const response = await fetch('/api/chats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: 'New Chat'
            })
        });
        
        if (response.ok) {
            const chat = await response.json();
            currentChatId = chat.id;
            
            // Add chat to sidebar without reloading
            addChatToSidebar(chat);
            
            // Select the new chat
            selectChat(chat.id);
            
            // Show messages area
            showMessagesArea();
            
            // Clear messages list for new chat
            const messagesList = document.getElementById('messagesList');
            if (messagesList) {
                messagesList.innerHTML = '';
            }
        } else {
            const error = await response.json();
            alert('Error creating chat: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error creating chat:', error);
        alert('Error creating chat: ' + error.message);
    }
}

function addChatToSidebar(chat) {
    const chatsList = document.getElementById('chatsList');
    const emptyState = document.querySelector('.empty-state');
    
    // Hide empty state if it exists
    if (emptyState) {
        emptyState.style.display = 'none';
    }
    
    // Create chat item element
    const chatItem = document.createElement('div');
    chatItem.className = 'chat-item active';
    chatItem.setAttribute('data-chat-id', chat.id);
    
    chatItem.innerHTML = `
        <i class="fas fa-comment chat-icon"></i>
        <div class="chat-info">
            <p class="chat-title">${escapeHtml(chat.title)}</p>
            <p class="chat-meta">${chat.messages ? chat.messages.length : 0} messages</p>
        </div>
        <button class="chat-delete" data-chat-id="${chat.id}" aria-label="Delete chat">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to sidebar (at the top)
    if (chatsList) {
        chatsList.insertBefore(chatItem, chatsList.firstChild);
    } else {
        // Create chats list if it doesn't exist
        const chatsView = document.getElementById('chatsView');
        if (chatsView) {
            const newChatsList = document.createElement('div');
            newChatsList.className = 'chats-list';
            newChatsList.id = 'chatsList';
            newChatsList.appendChild(chatItem);
            chatsView.appendChild(newChatsList);
        }
    }
    
    // Remove active class from other chats
    document.querySelectorAll('.chat-item').forEach(item => {
        if (item.dataset.chatId !== chat.id) {
            item.classList.remove('active');
        }
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Chat Search Functionality
const chatSearchInput = document.getElementById('chatSearchInput');
if (chatSearchInput) {
    chatSearchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase().trim();
        const chatItems = document.querySelectorAll('.chat-item');
        
        chatItems.forEach(item => {
            const chatTitle = item.querySelector('.chat-title')?.textContent.toLowerCase() || '';
            const chatMeta = item.querySelector('.chat-meta')?.textContent.toLowerCase() || '';
            
            if (searchTerm === '' || chatTitle.includes(searchTerm) || chatMeta.includes(searchTerm)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
        
        // Show empty state if no results
        const visibleChats = Array.from(chatItems).filter(item => item.style.display !== 'none');
        const emptyState = document.querySelector('.empty-state');
        if (emptyState) {
            if (visibleChats.length === 0 && searchTerm !== '') {
                emptyState.style.display = 'block';
                emptyState.innerHTML = `
                    <i class="fas fa-search empty-icon"></i>
                    <p>No chats found</p>
                    <p class="empty-subtitle">Try a different search term</p>
                `;
            } else if (searchTerm === '') {
                emptyState.style.display = visibleChats.length === 0 ? 'block' : 'none';
            } else {
                emptyState.style.display = 'none';
            }
        }
    });
}

// Chat selection and deletion
document.addEventListener('click', async (e) => {
    // Handle chat deletion
    if (e.target.closest('.chat-delete') || e.target.closest('.chat-delete i')) {
        const deleteBtn = e.target.closest('.chat-delete') || e.target.closest('.chat-delete').parentElement;
        const chatId = deleteBtn.dataset.chatId || deleteBtn.closest('[data-chat-id]')?.dataset.chatId;
        if (chatId) {
            e.stopPropagation();
            e.preventDefault();
            await deleteChat(chatId);
            return;
        }
    }
    
    // Handle chat selection (only if not clicking delete button)
    if (e.target.closest('.chat-item') && !e.target.closest('.chat-delete')) {
        const chatItem = e.target.closest('.chat-item');
        const chatId = chatItem.dataset.chatId;
        if (chatId) {
            selectChat(chatId);
        }
    }
});

function selectChat(chatId) {
    currentChatId = chatId;
    
    // Update active state
    document.querySelectorAll('.chat-item').forEach(item => {
        item.classList.toggle('active', item.dataset.chatId === chatId);
    });
    
    // Load chat messages
    loadChatMessages(chatId);
    
    // Show analytics buttons immediately
    const analyticsButtons = document.getElementById('analyticsButtons');
    if (analyticsButtons) {
        analyticsButtons.style.display = 'flex';
    }
}

async function deleteChat(chatId) {
    if (!confirm('Are you sure you want to delete this chat?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/chats/${chatId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Remove from UI immediately
            const chatItem = document.querySelector(`[data-chat-id="${chatId}"]`);
            if (chatItem) {
                chatItem.style.transition = 'opacity 0.3s';
                chatItem.style.opacity = '0';
                setTimeout(() => {
                    chatItem.remove();
                    // If deleted chat was active, show welcome screen
                    if (currentChatId === chatId) {
                        showWelcomeScreen();
                    }
                    // Reload if no chats left
                    const remainingChats = document.querySelectorAll('.chat-item');
                    if (remainingChats.length === 0) {
                        window.location.reload();
                    }
                }, 300);
            } else {
                // Fallback to reload
                window.location.reload();
            }
        } else {
            const error = await response.json();
            alert('Error deleting chat: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error deleting chat:', error);
        alert('Error deleting chat: ' + error.message);
    }
}

async function loadChatMessages(chatId) {
    try {
        const response = await fetch(`/api/chats/${chatId}/messages`);
        if (response.ok) {
            const messages = await response.json();
            displayMessages(messages);
            showMessagesArea();
        } else {
            console.error('Error loading messages:', await response.text());
        }
    } catch (error) {
        console.error('Error loading messages:', error);
    }
}

function displayMessages(messages) {
    const messagesList = document.getElementById('messagesList');
    messagesList.innerHTML = '';
    
    messages.forEach(message => {
        const messageEl = createMessageElement(message);
        messagesList.appendChild(messageEl);
    });
    
    messagesList.scrollTop = messagesList.scrollHeight;
}

function createMessageElement(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;
    if (message.id && message.id.startsWith('temp-')) {
        messageDiv.setAttribute('data-temp-id', message.id);
    }
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = message.role === 'user' 
        ? '<i class="fas fa-user"></i>'
        : '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = message.content;
    
    // Add sources if available
    if (message.sources && message.sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.style.fontSize = '0.75rem';
        sourcesDiv.style.marginTop = '0.5rem';
        sourcesDiv.style.opacity = '0.7';
        sourcesDiv.innerHTML = '<strong>Sources:</strong> ' + message.sources.join(', ');
        content.appendChild(sourcesDiv);
    }
    
    // Add scoring box for assistant messages (confidence & completeness scores)
    if (message.role === 'assistant') {
        const scoringDiv = document.createElement('div');
        scoringDiv.className = 'message-scoring';
        scoringDiv.style.cssText = `
            margin-top: 0.75rem;
            padding: 0.5rem 0.75rem;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 0.375rem;
            font-size: 0.75rem;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        `;
        
        let scoringHTML = '<div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">';
        
        // Confidence Score
        if (message.confidence_score !== undefined && message.confidence_score !== null) {
            const confidencePercent = (message.confidence_score * 100).toFixed(1);
            const confidenceColor = message.confidence_score >= 0.7 ? '#10B981' : message.confidence_score >= 0.4 ? '#F59E0B' : '#EF4444';
            scoringHTML += `
                <div style="display: flex; align-items: center; gap: 0.25rem;">
                    <strong style="color: #3B82F6;">Confidence:</strong>
                    <span style="color: ${confidenceColor}; font-weight: 600;">${confidencePercent}%</span>
                </div>
            `;
        } else {
            scoringHTML += `
                <div style="display: flex; align-items: center; gap: 0.25rem;">
                    <strong style="color: #3B82F6;">Confidence:</strong>
                    <span style="color: #9CA3AF;">N/A</span>
                </div>
            `;
        }
        
        // Completeness Score
        if (message.completeness_score !== undefined && message.completeness_score !== null) {
            const completenessPercent = (message.completeness_score * 100).toFixed(1);
            const completenessColor = message.completeness_score >= 0.7 ? '#10B981' : message.completeness_score >= 0.4 ? '#F59E0B' : '#EF4444';
            scoringHTML += `
                <div style="display: flex; align-items: center; gap: 0.25rem;">
                    <strong style="color: #3B82F6;">Completeness:</strong>
                    <span style="color: ${completenessColor}; font-weight: 600;">${completenessPercent}%</span>
                </div>
            `;
        } else {
            scoringHTML += `
                <div style="display: flex; align-items: center; gap: 0.25rem;">
                    <strong style="color: #3B82F6;">Completeness:</strong>
                    <span style="color: #9CA3AF;">N/A</span>
                </div>
            `;
        }
        
        scoringHTML += '</div>';
        scoringDiv.innerHTML = scoringHTML;
        content.appendChild(scoringDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    return messageDiv;
}

function showMessagesArea() {
    document.getElementById('welcomeScreen').classList.add('hidden');
    document.getElementById('messagesArea').classList.remove('hidden');
    
    // Show analytics buttons when messages area is shown
    const analyticsButtons = document.getElementById('analyticsButtons');
    if (analyticsButtons && currentChatId) {
        analyticsButtons.style.display = 'flex';
    }
}

function showWelcomeScreen() {
    document.getElementById('welcomeScreen').classList.remove('hidden');
    document.getElementById('messagesArea').classList.add('hidden');
    currentChatId = null;
    
    // Hide analytics buttons
    hideAnalyticsButtons();
}

// Message sending
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const voiceIcon = document.getElementById('voiceIcon');
const voiceStatus = document.getElementById('voiceStatus');
const voiceStatusText = document.getElementById('voiceStatusText');

// Voice Recognition
let recognition = null;
let isRecording = false;

// Check if browser supports speech recognition
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
        isRecording = true;
        voiceBtn.classList.add('recording');
        voiceIcon.className = 'fas fa-stop';
        voiceStatus.style.display = 'block';
        voiceStatus.classList.add('recording');
        voiceStatusText.textContent = 'Listening...';
    };
    
    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Update input field with interim results
        messageInput.value = finalTranscript + interimTranscript;
        
        // If we have final transcript, stop and send
        if (finalTranscript.trim()) {
            recognition.stop();
        }
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopVoiceRecognition();
        
        let errorMsg = 'Error: ';
        switch(event.error) {
            case 'no-speech':
                errorMsg = 'No speech detected. Please try again.';
                break;
            case 'audio-capture':
                errorMsg = 'Microphone not found. Please check your microphone.';
                break;
            case 'not-allowed':
                errorMsg = 'Microphone permission denied. Please allow microphone access.';
                break;
            default:
                errorMsg = 'Speech recognition error. Please try again.';
        }
        
        voiceStatusText.textContent = errorMsg;
        setTimeout(() => {
            voiceStatus.style.display = 'none';
        }, 3000);
    };
    
    recognition.onend = () => {
        stopVoiceRecognition();
        
        // If there's text in the input, automatically send it
        if (messageInput.value.trim()) {
            // Small delay to show the text was captured
            setTimeout(() => {
                sendMessage();
            }, 300);
        }
    };
    
    function stopVoiceRecognition() {
        isRecording = false;
        voiceBtn.classList.remove('recording');
        voiceIcon.className = 'fas fa-microphone';
        voiceStatus.style.display = 'none';
        voiceStatus.classList.remove('recording');
        
        if (recognition && recognition.state !== 'inactive') {
            recognition.stop();
        }
    }
    
    // Voice button click handler
    if (voiceBtn) {
        voiceBtn.addEventListener('click', () => {
            if (isRecording) {
                // Stop recording
                stopVoiceRecognition();
            } else {
                // Start recording
                try {
                    recognition.start();
                } catch (error) {
                    console.error('Error starting recognition:', error);
                    voiceStatus.style.display = 'block';
                    voiceStatusText.textContent = 'Error starting voice recognition. Please try again.';
                    setTimeout(() => {
                        voiceStatus.style.display = 'none';
                    }, 3000);
                }
            }
        });
    }
} else {
    // Browser doesn't support speech recognition
    if (voiceBtn) {
        voiceBtn.style.display = 'none';
    }
    console.warn('Speech recognition not supported in this browser');
}

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content) return;
    
    // If no chat selected, create a new one
    if (!currentChatId) {
        try {
            const response = await fetch('/api/chats', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: 'New Chat'
                })
            });
            
            if (response.ok) {
                const chat = await response.json();
                currentChatId = chat.id;
                
                // Add chat to sidebar
                addChatToSidebar(chat);
                
                // Select the chat in sidebar
                selectChat(currentChatId);
                
                // Show messages area
                showMessagesArea();
            } else {
                alert('Error creating chat');
                return;
            }
        } catch (error) {
            console.error('Error creating chat:', error);
            alert('Error creating chat');
            return;
        }
    }
    
    // Clear input immediately for better UX
    messageInput.value = '';
    
    // Show user message immediately
    const userMsg = {
        id: 'temp-' + Date.now(),
        content: content,
        role: 'user',
        timestamp: new Date().toISOString()
    };
    const messagesList = document.getElementById('messagesList');
    if (messagesList) {
        messagesList.appendChild(createMessageElement(userMsg));
        messagesList.scrollTop = messagesList.scrollHeight;
    }
    
    try {
        // Send to backend - it will save user message and get AI response
        const response = await fetch(`/api/chats/${currentChatId}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content: content,
                role: 'user'
            })
        });
        
        if (response.ok) {
            // Get the AI response first (can only read response once!)
            const aiMessage = await response.json();
            
            // Update stats if provided
            if (aiMessage.statistics) {
                updateStatisticsDisplay(aiMessage.statistics);
            }
            
            // Update chat title if changed (this happens when first message is sent)
            if (aiMessage.chat_title) {
                updateChatTitle(currentChatId, aiMessage.chat_title);
            }
            
            // Ensure chat is visible in sidebar (in case it wasn't added yet)
            const chatItem = document.querySelector(`[data-chat-id="${currentChatId}"]`);
            if (!chatItem) {
                // Chat not in sidebar, fetch and add it
                try {
                    const chatResponse = await fetch(`/api/chats/${currentChatId}`);
                    if (chatResponse.ok) {
                        const chat = await chatResponse.json();
                        addChatToSidebar(chat);
                    }
                } catch (error) {
                    console.error('Error fetching chat:', error);
                }
            }
            
            // Show AI response immediately
            if (messagesList) {
                const aiMsgElement = createMessageElement({
                    id: aiMessage.id || 'temp-ai-' + Date.now(),
                    content: aiMessage.content,
                    role: 'assistant',
                    timestamp: aiMessage.timestamp || new Date().toISOString(),
                    sources: aiMessage.sources || [],
                    confidence_score: aiMessage.confidence_score,
                    completeness_score: aiMessage.completeness_score
                });
                messagesList.appendChild(aiMsgElement);
                messagesList.scrollTop = messagesList.scrollHeight;
            }
            
            // Remove temporary user message
            if (messagesList) {
                const tempMsg = messagesList.querySelector(`[data-temp-id="${userMsg.id}"]`);
                if (tempMsg) tempMsg.remove();
            }
            
            // Reload all messages from database to get saved versions (this will replace temp messages)
            setTimeout(async () => {
                await loadChatMessages(currentChatId);
            }, 100);
            
            // Show recommendations if available
            if (aiMessage.recommendations && aiMessage.recommendations.similar_queries.length > 0) {
                console.log('Recommendations:', aiMessage.recommendations);
            }
        } else {
            // Handle HTTP errors
            let errorMessage = 'Server error occurred. ';
            try {
                const error = await response.json();
                errorMessage += error.error || 'Failed to get response from server.';
                console.error('Server error:', error);
            } catch (parseError) {
                errorMessage += `HTTP ${response.status}: ${response.statusText || 'Unknown error'}`;
                console.error('Error parsing response:', parseError);
            }
            
            alert(errorMessage);
            
            // Restore the input text so user can try again
            messageInput.value = content;
            
            // Remove the temporary user message on error
            if (messagesList) {
                const tempMsg = messagesList.querySelector(`[data-temp-id="${userMsg.id}"]`);
                if (tempMsg) tempMsg.remove();
            }
        }
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Better error handling
        let errorMessage = 'Unable to connect to server. ';
        if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
            errorMessage += 'Please check if the server is running and try again.';
        } else if (error.message) {
            errorMessage += error.message;
        } else {
            errorMessage += 'Please try again later.';
        }
        
        // Show user-friendly error
        alert(errorMessage);
        
        // Restore the input text so user can try again
        messageInput.value = content;
        
        // Remove the temporary user message on error
        const messagesList = document.getElementById('messagesList');
        if (messagesList) {
            const tempMsg = messagesList.querySelector(`[data-temp-id="${userMsg.id}"]`);
            if (tempMsg) tempMsg.remove();
        }
    }
}

// Upload button - PDF upload functionality (only inline button now)
const uploadBtnSmall = document.getElementById('uploadBtnSmall');
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.pdf';
fileInput.style.display = 'none';
document.body.appendChild(fileInput);

// Function to handle file upload
async function handleFileUpload(file) {
    if (!file) return;
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Please select a PDF file');
        return;
    }
    
    // Show loading state
    if (uploadBtnSmall) {
        uploadBtnSmall.disabled = true;
        uploadBtnSmall.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    }
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        // Include current chat_id if available, so PDF is associated with current chat
        if (currentChatId) {
            formData.append('chat_id', currentChatId);
        }
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            alert(`PDF uploaded successfully!\n\nFile: ${result.filename}\nPages: ${result.page_count}\nProcessed: ${result.processed ? 'Yes' : 'No'}`);
            
            // If in a chat, reload to show updated context
            if (currentChatId) {
                await loadChatMessages(currentChatId);
            }
        } else {
            let errorMessage = 'Error uploading PDF. ';
            try {
                const error = await response.json();
                errorMessage += error.error || 'Unknown error';
            } catch (parseError) {
                errorMessage += `HTTP ${response.status}: ${response.statusText || 'Unknown error'}`;
            }
            alert(errorMessage);
        }
    } catch (error) {
        console.error('Upload error:', error);
        let errorMessage = 'Error uploading PDF. ';
        if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
            errorMessage += 'Please check if the server is running and try again.';
        } else {
            errorMessage += error.message || 'Please try again later.';
        }
        alert(errorMessage);
    } finally {
        // Reset button states
        if (uploadBtnSmall) {
            uploadBtnSmall.disabled = false;
            uploadBtnSmall.innerHTML = '<i class="fas fa-file-pdf"></i>';
        }
        // Reset file input
        fileInput.value = '';
    }
}

// Upload button in input area
if (uploadBtnSmall) {
    uploadBtnSmall.addEventListener('click', () => {
        fileInput.click();
    });
}

// File input change handler
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    await handleFileUpload(file);
});

// Add file input to body
document.body.appendChild(fileInput);

// Update statistics display in real-time
function updateStatisticsDisplay(stats) {
    const statTotalChats = document.getElementById('statTotalChats');
    const statTotalMessages = document.getElementById('statTotalMessages');
    const statAvgTime = document.getElementById('statAvgTime');
    const statVectorSearches = document.getElementById('statVectorSearches');
    const statGraphSearches = document.getElementById('statGraphSearches');
    
    if (statTotalChats) statTotalChats.textContent = stats.total_chats || 0;
    if (statTotalMessages) statTotalMessages.textContent = stats.total_messages || 0;
    if (statAvgTime) statAvgTime.textContent = (stats.average_response_time || 0) + 's';
    if (statVectorSearches) statVectorSearches.textContent = stats.vector_searches || 0;
    if (statGraphSearches) statGraphSearches.textContent = stats.graph_searches || 0;
}

// Update chat title in sidebar
function updateChatTitle(chatId, newTitle) {
    const chatItem = document.querySelector(`[data-chat-id="${chatId}"]`);
    if (chatItem) {
        const titleElement = chatItem.querySelector('.chat-title');
        if (titleElement) {
            titleElement.textContent = newTitle;
            // Also update message count if needed
            const metaElement = chatItem.querySelector('.chat-meta');
            if (metaElement) {
                // Get current message count and update
                const messagesList = document.getElementById('messagesList');
                if (messagesList) {
                    const messageCount = messagesList.querySelectorAll('.message').length;
                    metaElement.textContent = `${messageCount} messages`;
                }
            }
        }
    }
}

// Analytics Functions
function showAnalyticsButtons() {
    const analyticsButtons = document.getElementById('analyticsButtons');
    if (analyticsButtons && currentChatId) {
        analyticsButtons.style.display = 'flex';
    } else if (analyticsButtons) {
        analyticsButtons.style.display = 'none';
    }
}

function hideAnalyticsButtons() {
    const analyticsButtons = document.getElementById('analyticsButtons');
    if (analyticsButtons) {
        analyticsButtons.style.display = 'none';
    }
}

function showAnalyticsModal(title, content) {
    const modal = document.getElementById('analyticsModal');
    const modalTitle = document.getElementById('analyticsModalTitle');
    const modalBody = document.getElementById('analyticsModalBody');
    
    if (modal && modalTitle && modalBody) {
        modalTitle.textContent = title;
        modalBody.innerHTML = content;
        modal.classList.add('active');
    }
}

function closeAnalyticsModal() {
    const modal = document.getElementById('analyticsModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// Close modal on outside click
document.addEventListener('click', (e) => {
    const modal = document.getElementById('analyticsModal');
    if (modal && e.target === modal) {
        closeAnalyticsModal();
    }
});

// Close modal button
const analyticsModalClose = document.getElementById('analyticsModalClose');
if (analyticsModalClose) {
    analyticsModalClose.addEventListener('click', closeAnalyticsModal);
}

// Patterns Button
const patternsBtn = document.getElementById('patternsBtn');
if (patternsBtn) {
    patternsBtn.addEventListener('click', async () => {
        if (!currentChatId) {
            alert('Please select a chat first');
            return;
        }
        
        try {
            patternsBtn.disabled = true;
            patternsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Loading...</span>';
            
            const response = await fetch(`/api/patterns/${currentChatId}`);
            const data = await response.json();
            
            let content = '';
            if (data.patterns && data.patterns.length > 0) {
                content = '<div class="analytics-item"><div class="analytics-item-title">Summary</div><div class="analytics-item-content">' + data.message + '</div></div>';
                data.patterns.forEach((pattern, index) => {
                    content += `
                        <div class="analytics-item">
                            <div class="analytics-item-title">${index + 1}. ${pattern.type.replace(/_/g, ' ').toUpperCase()}</div>
                            <div class="analytics-item-content">${pattern.description}</div>
                            <div class="analytics-item-meta">Frequency: ${pattern.frequency} time${pattern.frequency > 1 ? 's' : ''}</div>
                            ${pattern.examples ? `<div class="analytics-item-meta">Examples: ${pattern.examples.slice(0, 2).join(', ')}</div>` : ''}
                        </div>
                    `;
                });
            } else {
                content = `<div class="analytics-empty">${data.message || 'No patterns found'}</div>`;
            }
            
            showAnalyticsModal('Query Patterns', content);
        } catch (error) {
            console.error('Error fetching patterns:', error);
            alert('Error loading patterns: ' + error.message);
        } finally {
            patternsBtn.disabled = false;
            patternsBtn.innerHTML = '<i class="fas fa-project-diagram"></i> <span>Patterns</span>';
        }
    });
}

// Anomalies Button
const anomaliesBtn = document.getElementById('anomaliesBtn');
if (anomaliesBtn) {
    anomaliesBtn.addEventListener('click', async () => {
        if (!currentChatId) {
            alert('Please select a chat first');
            return;
        }
        
        try {
            anomaliesBtn.disabled = true;
            anomaliesBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Loading...</span>';
            
            const response = await fetch(`/api/anomalies/${currentChatId}`);
            const data = await response.json();
            
            let content = '';
            if (data.anomalies && data.anomalies.length > 0) {
                content = '<div class="analytics-item"><div class="analytics-item-title">Summary</div><div class="analytics-item-content">' + data.message + '</div></div>';
                data.anomalies.forEach((anomaly, index) => {
                    content += `
                        <div class="analytics-item">
                            <div class="analytics-item-title">${index + 1}. ${anomaly.type.replace(/_/g, ' ').toUpperCase()}</div>
                            <div class="analytics-item-content">${anomaly.description}</div>
                            <div class="analytics-item-meta">Query: "${anomaly.query}"</div>
                            ${anomaly.confidence !== undefined ? `<div class="analytics-item-meta">Confidence: ${(anomaly.confidence * 100).toFixed(1)}%</div>` : ''}
                        </div>
                    `;
                });
            } else {
                content = `<div class="analytics-empty">${data.message || 'No anomalies found'}</div>`;
            }
            
            showAnalyticsModal('Detected Anomalies', content);
        } catch (error) {
            console.error('Error fetching anomalies:', error);
            alert('Error loading anomalies: ' + error.message);
        } finally {
            anomaliesBtn.disabled = false;
            anomaliesBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Anomalies</span>';
        }
    });
}

// Trends Button
const trendsBtn = document.getElementById('trendsBtn');
if (trendsBtn) {
    trendsBtn.addEventListener('click', async () => {
        if (!currentChatId) {
            alert('Please select a chat first');
            return;
        }
        
        try {
            trendsBtn.disabled = true;
            trendsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Loading...</span>';
            
            const response = await fetch(`/api/trends/${currentChatId}?days=30`);
            const data = await response.json();
            
            let content = '';
            if (data.trends) {
                const trends = data.trends;
                
                // Graph Query Frequency
                if (trends.graph_query_frequency) {
                    content += '<div class="analytics-item"><div class="analytics-item-title">Graph Query Frequency</div>';
                    content += `<div class="analytics-item-content">Vector: ${trends.graph_query_frequency.total_vector || 0} | Graph: ${trends.graph_query_frequency.total_graph || 0} | Hybrid: ${trends.graph_query_frequency.total_hybrid || 0}</div></div>`;
                }
                
                // Rising Topics
                if (trends.rising_topics && trends.rising_topics.length > 0) {
                    content += '<div class="analytics-item"><div class="analytics-item-title">Rising Topics</div>';
                    trends.rising_topics.forEach(topic => {
                        content += `<div class="analytics-item-content">${topic.topic}: ${topic.growth_rate}% growth (${topic.recent_count} recent vs ${topic.early_count} early)</div>`;
                    });
                    content += '</div>';
                }
                
                // Embedding Trends
                if (trends.embedding_trends && trends.embedding_trends.length > 0) {
                    content += '<div class="analytics-item"><div class="analytics-item-title">Embedding Trends</div>';
                    trends.embedding_trends.slice(-5).forEach(trend => {
                        content += `<div class="analytics-item-content">${trend.date}: ${trend.query_count} queries</div>`;
                    });
                    content += '</div>';
                }
                
                if (!content) {
                    content = '<div class="analytics-empty">No trend data available</div>';
                }
            } else {
                content = '<div class="analytics-empty">No trends found</div>';
            }
            
            showAnalyticsModal('Trend Analysis', content);
        } catch (error) {
            console.error('Error fetching trends:', error);
            alert('Error loading trends: ' + error.message);
        } finally {
            trendsBtn.disabled = false;
            trendsBtn.innerHTML = '<i class="fas fa-chart-line"></i> <span>Trends</span>';
        }
    });
}

// Recommendations Button
const recommendationsBtn = document.getElementById('recommendationsBtn');
if (recommendationsBtn) {
    recommendationsBtn.addEventListener('click', async () => {
        if (!currentChatId) {
            alert('Please select a chat first');
            return;
        }
        
        try {
            recommendationsBtn.disabled = true;
            recommendationsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Loading...</span>';
            
            const response = await fetch(`/api/recommendations/${currentChatId}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch recommendations');
            }
            
            let content = '';
            const recs = data.recommendations || {};
            
            // Similar Queries
            if (recs.similar_queries && recs.similar_queries.length > 0) {
                content += '<div class="analytics-item"><div class="analytics-item-title">Similar Queries</div>';
                recs.similar_queries.forEach((query, index) => {
                    content += `<div class="analytics-item-content">${index + 1}. "${query}"</div>`;
                });
                content += '</div>';
            }
            
            // Similar Documents
            if (recs.similar_documents && recs.similar_documents.length > 0) {
                content += '<div class="analytics-item"><div class="analytics-item-title">Similar Documents</div>';
                recs.similar_documents.forEach((doc, index) => {
                    const source = doc.source === 'pdf' ? `PDF: ${doc.filename || 'Document'}` : 'Vector DB';
                    const similarity = (doc.similarity * 100).toFixed(1);
                    content += `<div class="analytics-item-content">${index + 1}. ${source} (${similarity}% similar)</div>`;
                    if (doc.content) {
                        content += `<div class="analytics-item-meta">${doc.content}</div>`;
                    }
                });
                content += '</div>';
            }
            
            // Connected Nodes
            if (recs.connected_nodes && recs.connected_nodes.length > 0) {
                content += '<div class="analytics-item"><div class="analytics-item-title">Connected Graph Nodes</div>';
                recs.connected_nodes.forEach((node, index) => {
                    content += `<div class="analytics-item-content">${index + 1}. ${node.name} (${node.type})</div>`;
                    if (node.source_node) {
                        content += `<div class="analytics-item-meta">Connected to: ${node.source_node}</div>`;
                    }
                });
                content += '</div>';
            }
            
            // Related Concepts
            if (recs.related_concepts && recs.related_concepts.length > 0) {
                content += '<div class="analytics-item"><div class="analytics-item-title">Related Concepts</div>';
                content += `<div class="analytics-item-content">${recs.related_concepts.join(', ')}</div>`;
                content += '</div>';
            }
            
            // Suggested Topics
            if (recs.suggested_topics && recs.suggested_topics.length > 0) {
                content += '<div class="analytics-item"><div class="analytics-item-title">Suggested Topics</div>';
                content += `<div class="analytics-item-content">${recs.suggested_topics.join(', ')}</div>`;
                content += '</div>';
            }
            
            if (!content) {
                content = '<div class="analytics-empty">No recommendations available. Send a message first to generate recommendations.</div>';
            } else {
                content = `<div class="analytics-item"><div class="analytics-item-title">Query</div><div class="analytics-item-content">"${data.query || 'Last query'}"</div></div>` + content;
            }
            
            showAnalyticsModal('Recommendations', content);
        } catch (error) {
            console.error('Error loading recommendations:', error);
            alert('Error loading recommendations: ' + error.message);
        } finally {
            recommendationsBtn.disabled = false;
            recommendationsBtn.innerHTML = '<i class="fas fa-lightbulb"></i> <span>Recommendations</span>';
        }
    });
}

// Analytics buttons are now shown/hidden in selectChat and showWelcomeScreen functions above

// Initialize on load
initTheme();
switchView('chats');