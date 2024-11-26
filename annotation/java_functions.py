

def return_java_function(java_input):
    if java_input == 'zoom_in':
        return """
            () => {
                let images = document.querySelectorAll('img');
                images.forEach(img => {
                    let scale = img.style.transform ? parseFloat(img.style.transform.match(/scale\\((.*?)\\)/)?.[1] || 1) : 1;
                    scale = scale * 1.2;  // Zoom in by 1.2x
                    img.style.transform = `scale(${scale})`;
                });
            }
        """
    elif java_input == 'zoom_out':
        return """
            () => {
                let images = document.querySelectorAll('img');
                images.forEach(img => {
                    let scale = img.style.transform ? parseFloat(img.style.transform.match(/scale\\((.*?)\\)/)?.[1] || 1) : 1;
                    scale = scale / 1.2;  // Zoom out by dividing scale by 1.2
                    img.style.transform = `scale(${scale})`;
                });
            }
        """
    elif java_input == 'reset_zoom':
        return """
            () => {
                let images = document.querySelectorAll('img');
                images.forEach(img => {
                    img.style.transform = 'scale(1)';
                });
            }
        """
    elif java_input == 'image_dragging':
        return """
            () => {
                function addDragToImage() {
                    const images = document.querySelectorAll('img');
                    images.forEach(img => {
                        let isDragging = false;
                        let startX, startY;
                        let translateX = 0;
                        let translateY = 0;

                        // Prevent context menu
                        img.addEventListener('contextmenu', (e) => {
                            e.preventDefault();
                        });

                        // Start drag on right mouse down
                        img.addEventListener('mousedown', (e) => {
                            if (e.button === 2) { // Right mouse button
                                e.preventDefault();
                                isDragging = true;
                                startX = e.clientX - translateX;
                                startY = e.clientY - translateY;
                            }
                        });

                        // Handle drag
                        document.addEventListener('mousemove', (e) => {
                            if (!isDragging) return;

                            translateX = e.clientX - startX;
                            translateY = e.clientY - startY;

                            const scale = img.style.transform.match(/scale\\((.*?)\\)/) 
                                ? parseFloat(img.style.transform.match(/scale\\((.*?)\\)/)[1]) 
                                : 1;

                            img.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
                        });

                        // Stop drag
                        document.addEventListener('mouseup', (e) => {
                            if (e.button === 2) { // Right mouse button
                                isDragging = false;
                            }
                        });
                    });
                }

                // Initial setup
                addDragToImage();

                // Re-apply drag functionality when image changes
                const observer = new MutationObserver(() => {
                    addDragToImage();
                });

                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
            }
        """
    elif java_input == 'keyboard_shortcuts':
        return """
            () => {
                document.addEventListener('keydown', (e) => {
                    if (e.ctrlKey && e.key === '+') {
                        document.querySelector('button:has(span:contains("🔍 Zoom In"))').click();
                        e.preventDefault();
                    } else if (e.ctrlKey && e.key === '-') {
                        document.querySelector('button:has(span:contains("🔍 Zoom Out"))').click();
                        e.preventDefault();
                    } else if (e.ctrlKey && e.key === '0') {
                        document.querySelector('button:has(span:contains("↺ Reset View"))').click();
                        e.preventDefault();
                    } else if (e.key === 'ArrowLeft') {
                        // Find and click the Frame View previous button
                        document.querySelector('button:has(span:contains("Previous Frame"))').click();
                        e.preventDefault();
                    } else if (e.key === 'ArrowRight') {
                        // Find and click the Frame View next button
                        document.querySelector('button:has(span:contains("Next Frame"))').click();
                        e.preventDefault();
                    }
                });
            }
        """
    elif java_input == 'frame_per':
        return """
                            () => {
                                document.addEventListener('keydown', (e) => {
                                    let slider = document.querySelector('input[type="range"]');
                                    let step = 1.0;
                                    let currentValue = parseFloat(slider.value);

                                    if (e.key === 'ArrowLeft') {
                                        let newValue = Math.max(0, currentValue - step);
                                        slider.value = newValue;
                                        slider.dispatchEvent(new Event('input'));
                                        slider.dispatchEvent(new Event('change'));
                                    } else if (e.key === 'ArrowRight') {
                                        let newValue = Math.min(100, currentValue + step);
                                        slider.value = newValue;
                                        slider.dispatchEvent(new Event('input'));
                                        slider.dispatchEvent(new Event('change'));
                                    }
                                });
                                return [];
                            }
                        """