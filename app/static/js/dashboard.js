document.addEventListener('DOMContentLoaded', () => {
    // Navigation / Tab System
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const tabTitle = document.getElementById('tab-title');
    const tabSubtitle = document.getElementById('tab-subtitle');

    const tabMeta = {
        'dashboard': { title: 'Sistem Telemetrisi', subtitle: 'Yerel Edge Cihaz Güç ve Kaynak Tüketim Analizleri' },
        'swarm': { title: 'Sürü Harp Simülatörü', subtitle: 'Yerel Edge AI Kararları ile Otonom Sürü İletişimi' },
        'health': { title: 'Sağlık ve Tanı', subtitle: 'Onkolojik Tarama ve Akıllı İlaç Keşfi Edge Analizi' },
        'education': { title: 'Kişiselleştirilmiş Eğitim', subtitle: 'Yerel Metriklerle Müfredat Adaptasyonu ve T3AI Entegrasyonu' },
        'federated': { title: 'Dağıtık Öğrenme Konsolu', subtitle: 'Privacy-Preserving Ağırlık Birleştirme (FedAvg) Yönetimi' },
        'security': { title: 'Siber Güvenlik & Veri Tabanı', subtitle: 'PQC İmza Doğrulama, Byzantine Defansı ve Yerel SQLite Denetimi' },
        'robotics': { title: 'Otonom Robotik & EKF Seyrüsefer', subtitle: 'Extended Kalman Filter ile Gürültülü Sensör Kestirimi ve PQC Tünelleme' }
    };

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');
            
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${tabName}`).classList.add('active');
            
            tabTitle.textContent = tabMeta[tabName].title;
            tabSubtitle.textContent = tabMeta[tabName].subtitle;

            // Trigger canvas animation only in swarm tab
            if (tabName === 'swarm') {
                startSwarmSim();
            } else {
                stopSwarmSim();
            }

            if (tabName === 'security') {
                fetchSecurityAudit();
            }

            if (tabName === 'robotics') {
                fetchRoboticsStep();
            }
        });
    });

    // --- SYSTEM TELEMETRY POLLING ---
    function fetchTelemetry() {
        fetch('/api/telemetry')
            .then(res => res.json())
            .then(data => {
                document.getElementById('real-ram').textContent = data.real_process_ram_mb;
                document.getElementById('real-cpu').textContent = Math.round(data.cpu_percent);
                document.getElementById('power-nizam').textContent = data.nizam_edge_simulated_power_w;
                document.getElementById('power-traditional').textContent = data.traditional_server_power_w;
                
                document.getElementById('runtime-nizam').textContent = data.runtime_nizam_hours + ' sa';
                document.getElementById('runtime-traditional').textContent = data.runtime_traditional_hours + ' sa';
                
                // Update battery display card value
                document.getElementById('battery-hours').textContent = data.runtime_nizam_hours;
            })
            .catch(err => console.error("Telemetry fetch error:", err));
    }
    
    // Poll telemetry every 3 seconds
    fetchTelemetry();
    setInterval(fetchTelemetry, 3000);


    // --- SWARM SIMULATION TACTICAL CANVAS ---
    const canvas = document.getElementById('swarm-canvas');
    const ctx = canvas.getContext('2d');
    let swarmInterval = null;
    let isJamming = false;

    function startSwarmSim() {
        if (swarmInterval) return;
        swarmInterval = setInterval(updateSwarmSimulation, 100); // 10 FPS
    }

    function stopSwarmSim() {
        if (swarmInterval) {
            clearInterval(swarmInterval);
            swarmInterval = null;
        }
    }

    function updateSwarmSimulation() {
        fetch('/api/swarm/step')
            .then(res => res.json())
            .then(data => {
                drawSwarm(data);
                updateSwarmUI(data);
            })
            .catch(err => console.error("Swarm update error:", err));
    }

    function drawSwarm(data) {
        // Clear canvas
        ctx.fillStyle = '#0b0c14';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }

        // Draw Base (0, 0)
        ctx.strokeStyle = '#8e9bb3';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(20, 20, 10, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fillStyle = 'rgba(142, 155, 179, 0.2)';
        ctx.fill();

        // Draw Target Drone
        const target = data.target;
        ctx.shadowBlur = 15;
        ctx.shadowColor = '#f43f5e';
        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.arc(target.x, target.y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw sensor radius boundary for target
        ctx.shadowBlur = 0;
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.15)';
        ctx.beginPath();
        ctx.arc(target.x, target.y, 30, 0, 2 * Math.PI);
        ctx.stroke();

        // Draw Drones
        data.drones.forEach(drone => {
            // Velocity vector
            const isEvading = drone.action === 'Evade';
            const isCharging = drone.action === 'ReturnToBase';
            
            ctx.shadowBlur = 10;
            ctx.shadowColor = isEvading ? '#f59e0b' : isCharging ? '#a855f7' : '#00f2fe';
            ctx.fillStyle = isEvading ? '#f59e0b' : isCharging ? '#a855f7' : '#00f2fe';
            
            ctx.beginPath();
            ctx.arc(drone.x, drone.y, 6, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw connection line if close to target (assisting tracking)
            const dist = Math.hypot(target.x - drone.x, target.y - drone.y);
            if (dist < 120 && !data.jammed) {
                ctx.shadowBlur = 0;
                ctx.strokeStyle = 'rgba(0, 242, 254, 0.2)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(drone.x, drone.y);
                ctx.lineTo(target.x, target.y);
                ctx.stroke();
            }
        });
        
        // Reset shadow
        ctx.shadowBlur = 0;
    }

    function updateSwarmUI(data) {
        document.getElementById('swarm-status').textContent = data.jammed ? 'Karıştırma Engeli (Kaçış Modu)' : 'Aktif Kuşatma';
        document.getElementById('swarm-status').className = data.jammed ? 'text-amber' : 'text-cyan';
        
        document.getElementById('swarm-energy-nizam').textContent = data.summary_telemetry.total_nizam_energy_joules.toFixed(5) + ' J';
        document.getElementById('swarm-energy-trad').textContent = data.summary_telemetry.total_traditional_energy_joules.toFixed(5) + ' J';
        document.getElementById('swarm-gain').textContent = data.summary_telemetry.efficiency_gain_ratio + 'x';
        
        // Toggle Jamming btn active status
        const jamBtn = document.getElementById('toggle-jamming-btn');
        if (data.jammed) {
            jamBtn.classList.add('active');
            jamBtn.innerHTML = '<i class="fa-solid fa-tower-broadcast"></i> Elektronik Karıştırmayı Kapat';
        } else {
            jamBtn.classList.remove('active');
            jamBtn.innerHTML = '<i class="fa-solid fa-tower-broadcast"></i> Elektronik Karıştırma Başlat';
        }

        // Populate drone statuses table/list
        const listContainer = document.getElementById('drone-statuses');
        listContainer.innerHTML = '';
        data.drones.forEach(drone => {
            const row = document.createElement('div');
            row.className = 'drone-row';
            
            const badgeClass = drone.action === 'Evade' ? 'text-amber' : drone.action === 'ReturnToBase' ? 'text-violet' : 'text-cyan';
            
            row.innerHTML = `
                <span>Nizam-UAV-${drone.drone_id}</span>
                <span class="${badgeClass}">[${drone.action}]</span>
                <span>Pil: <strong>${drone.battery}%</strong></span>
            `;
            listContainer.appendChild(row);
        });
    }

    // Toggle jamming action
    document.getElementById('toggle-jamming-btn').addEventListener('click', () => {
        fetch('/api/swarm/toggle-jamming', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                isJamming = data.jamming_active;
            })
            .catch(err => console.error("Toggle jamming error:", err));
    });


    // --- HEALTH DIAGNOSTICS & DRUG SCREENING ---
    const healthForm = document.getElementById('health-form');
    const healthResultBox = document.getElementById('health-result-card');

    healthForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const area = document.getElementById('cell-area').value;
        const perimeter = document.getElementById('cell-perimeter').value;
        const texture = document.getElementById('cell-texture').value;
        const mitotic = document.getElementById('cell-mitotic').value;

        fetch('/api/diagnose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ area, perimeter, texture, mitotic })
        })
        .then(res => res.json())
        .then(data => {
            healthResultBox.classList.remove('hide');
            const outputEl = document.getElementById('diagnose-output');
            outputEl.textContent = data.diagnosis;
            
            if (data.diagnosis === 'Malignant') {
                outputEl.className = 'text-rose';
                healthResultBox.style.borderColor = 'rgba(244, 63, 94, 0.3)';
                healthResultBox.style.background = 'rgba(244, 63, 94, 0.08)';
            } else {
                outputEl.className = 'text-emerald';
                healthResultBox.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                healthResultBox.style.background = 'rgba(16, 185, 129, 0.08)';
            }
            
            document.getElementById('diagnose-conf').textContent = (data.confidence * 100).toFixed(0) + '%';
            
            const traditionalJ = data.telemetry.energy_traditional_joules;
            const nizamJ = data.telemetry.energy_nizam_joules;
            const savingsRatio = Math.round(traditionalJ / nizamJ);
            document.getElementById('diagnose-energy').textContent = savingsRatio.toLocaleString() + 'x Enerji Tasarrufu';
        })
        .catch(err => console.error("Diagnostics error:", err));
    });

    // Smart drug discovery click
    const screenDrugsBtn = document.getElementById('screen-drugs-btn');
    const drugsTableBody = document.querySelector('#drugs-table tbody');

    screenDrugsBtn.addEventListener('click', () => {
        drugsTableBody.innerHTML = '<tr><td colspan="4" class="text-center">Tarama yapılıyor (nicemlenmiş binary XNOR)...</td></tr>';
        
        fetch('/api/screen-drugs')
            .then(res => res.json())
            .then(data => {
                drugsTableBody.innerHTML = '';
                data.candidates.forEach(cand => {
                    const row = document.createElement('tr');
                    
                    const badgeClass = cand.potency_match === 'High' ? 'high' : cand.potency_match === 'Medium' ? 'medium' : 'low';
                    const badgeText = cand.potency_match === 'High' ? 'Yüksek' : cand.potency_match === 'Medium' ? 'Orta' : 'Düşük';
                    
                    row.innerHTML = `
                        <td><strong>${cand.compound_id}</strong></td>
                        <td>${cand.float_similarity}</td>
                        <td class="text-cyan">${cand.binary_similarity_score}</td>
                        <td><span class="badge-pill ${badgeClass}">${badgeText}</span></td>
                    `;
                    drugsTableBody.appendChild(row);
                });
            })
            .catch(err => {
                console.error("Drug screen error:", err);
                drugsTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-rose">Hata oluştu.</td></tr>';
            });
    });


    // --- PERSONALIZED EDUCATION ---
    // Update range labels on slide
    const ranges = ['std-attention', 'std-math', 'std-science', 'std-fatigue'];
    ranges.forEach(id => {
        const slider = document.getElementById(id);
        const label = document.getElementById(`${id}-val`);
        slider.addEventListener('input', () => {
            label.textContent = slider.value + (id.includes('attention') || id.includes('fatigue') ? '%' : '');
        });
    });

    const eduForm = document.getElementById('education-form');
    const chatDisplay = document.getElementById('chat-display');

    eduForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const attention = document.getElementById('std-attention').value;
        const math = document.getElementById('std-math').value;
        const science = document.getElementById('std-science').value;
        const fatigue = document.getElementById('std-fatigue').value;
        
        // Add user bubble showing state
        appendChatBubble(`Öğrenci Profili Değerlendiriliyor: Dikkat: ${attention}%, Mat: ${math}, Fen: ${science}, Yorgunluk: ${fatigue}%`, 'user');
        
        fetch('/api/education/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ attention, math, science, fatigue })
        })
        .then(res => res.json())
        .then(data => {
            const recommendedModeTR = data.recommended_mode === 'ChallengeMode' ? 'Zorlayıcı Mod (İleri Seviye)' : data.recommended_mode === 'ReinforceMode' ? 'Pekiştirme Modu (Orta Seviye)' : 'Tekrar Modu (Başlangıç Seviyesi)';
            
            let answer = `<strong>Tavsiye Edilen Eğitim Modu:</strong> <span class="text-emerald">${recommendedModeTR}</span> (Güven: ${(data.confidence*100).toFixed(0)}%)<br><br>`;
            answer += `<strong>Hedef Soru Şablonu:</strong> <em>"${data.original_question}"</em><br><br>`;
            answer += `<strong>T3AI Tarafından Üretilen Türkçe Açıklama:</strong><br>${data.t3ai_prompt_enriched || data.t3ai_enriched_prompt}`;
            
            appendChatBubble(answer, 'bot');
        })
        .catch(err => console.error("Education chat error:", err));
    });

    // Custom chat query
    const customChatInput = document.getElementById('custom-t3ai-input');
    const customChatBtn = document.getElementById('custom-t3ai-btn');

    function sendCustomT3AIQuery() {
        const text = customChatInput.value.trim();
        if (!text) return;
        
        appendChatBubble(text, 'user');
        customChatInput.value = '';
        
        fetch('/api/t3ai-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: text })
        })
        .then(res => res.json())
        .then(data => {
            appendChatBubble(data.choices[0].text, 'bot');
        })
        .catch(err => console.error("T3AI direct query error:", err));
    }

    customChatBtn.addEventListener('click', sendCustomT3AIQuery);
    customChatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendCustomT3AIQuery();
    });

    function appendChatBubble(content, sender) {
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${sender}`;
        bubble.innerHTML = `<p>${content}</p>`;
        chatDisplay.appendChild(bubble);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }


    // --- FEDERATED LEARNING ROUNDS ---
    const flRoundBtn = document.getElementById('fl-round-btn');
    const flRoundCount = document.getElementById('fl-round-count');
    const flWeightNorm = document.getElementById('fl-weight-norm');
    const flPrivacy = document.getElementById('fl-privacy');
    const socialPostsContainer = document.getElementById('social-posts');
    
    let currentRound = 0;

    flRoundBtn.addEventListener('click', () => {
        currentRound++;
        
        // Add animated updating status to node boxes
        const nodes = ['Tubitak-Node', 'Aselsan-Node', 'Havelsan-Node', 'Roketsan-Node'];
        nodes.forEach(nodeId => {
            const el = document.getElementById(`node-${nodeId}`);
            if (el) el.classList.add('updating');
        });
        
        flRoundBtn.disabled = true;
        flRoundBtn.textContent = 'FedAvg Parametreleri Toplanıyor...';
        
        fetch('/api/federated/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ round: currentRound })
        })
        .then(res => res.json())
        .then(data => {
            setTimeout(() => { // Add visual delay to show animation
                nodes.forEach(nodeId => {
                    const el = document.getElementById(`node-${nodeId}`);
                    if (el) el.classList.remove('updating');
                });
                
                flRoundBtn.disabled = false;
                flRoundBtn.innerHTML = '<i class="fa-solid fa-rotate"></i> Federated Averaging (FedAvg) Yuvarlağı Çalıştır';
                
                flRoundCount.textContent = currentRound;
                flWeightNorm.textContent = data.weights_norm;
                flPrivacy.textContent = data.privacy;
                
                // Update En Sosyal feed
                if (data.recent_social_posts && data.recent_social_posts.length > 0) {
                    socialPostsContainer.innerHTML = '';
                    data.recent_social_posts.reverse().forEach(post => {
                        const item = document.createElement('div');
                        item.className = 'post-item';
                        item.innerHTML = `
                            <div class="post-header">
                                <span class="post-author"><i class="fa-solid fa-seedling"></i> @${post.author}</span>
                                <span class="text-secondary">${new Date(post.timestamp * 1000).toLocaleTimeString()}</span>
                            </div>
                            <strong>${post.title}</strong>
                            <p>${post.content}</p>
                        `;
                        socialPostsContainer.appendChild(item);
                    });
                }
            }, 1200); // 1.2s delay for animation
        })
        .catch(err => {
            console.error("FL error:", err);
            nodes.forEach(nodeId => {
                const el = document.getElementById(`node-${nodeId}`);
                if (el) el.classList.remove('updating');
            });
            flRoundBtn.disabled = false;
            flRoundBtn.innerHTML = '<i class="fa-solid fa-rotate"></i> Hata Oluştu - Yeniden Dene';
        });
    });


    // --- KURE SEARCH DICTIONARY DIALOG ---
    const kureQueryInput = document.getElementById('kure-query-input');
    const kureSearchBtn = document.getElementById('kure-search-btn');
    const kureModal = document.getElementById('kure-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');

    function searchKure() {
        const text = kureQueryInput.value.trim();
        if (!text) return;
        
        fetch(`/api/query-kure?keyword=${encodeURIComponent(text)}`)
            .then(res => res.json())
            .then(data => {
                document.getElementById('modal-keyword').textContent = data.keyword;
                document.getElementById('modal-fact').textContent = data.fact;
                document.getElementById('modal-source').textContent = data.source;
                document.getElementById('modal-date').textContent = data.last_updated || 'Güncel Veri';
                
                kureModal.classList.remove('hide');
            })
            .catch(err => console.error("Küre query error:", err));
    }

    kureSearchBtn.addEventListener('click', searchKure);
    kureQueryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') searchKure();
    });

    closeModalBtn.addEventListener('click', () => {
        kureModal.classList.add('hide');
    });

    // Close modal if clicked outside content
    window.addEventListener('click', (e) => {
        if (e.target === kureModal) {
            kureModal.classList.add('hide');
        }
    });


    // --- SECURITY AUDIT & DATABASE LOGS ---
    function fetchSecurityAudit() {
        fetch('/api/security/audit')
            .then(res => res.json())
            .then(data => {
                const tbody = document.querySelector('#security-audit-table tbody');
                tbody.innerHTML = '';
                
                const totalRecords = (data.summary.telemetry_records + data.summary.federated_rounds + data.summary.health_diagnoses + data.summary.security_events);
                document.getElementById('db-records-count').textContent = totalRecords.toLocaleString();
                
                if (!data.audit_logs || data.audit_logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="text-center">Henüz olay kaydı yok.</td></tr>';
                    return;
                }

                data.audit_logs.forEach(log => {
                    const row = document.createElement('tr');
                    const statusClass = log.status === 'SUCCESS' || log.status === 'NORMAL' ? 'high' : log.status === 'REJECTED' ? 'low' : 'medium';
                    
                    row.innerHTML = `
                        <td>${log.timestamp}</td>
                        <td><strong>${log.event_type}</strong></td>
                        <td>${log.node_id}</td>
                        <td><span class="badge-pill ${statusClass}">${log.status}</span></td>
                        <td>${log.details}</td>
                    `;
                    tbody.appendChild(row);
                });
            })
            .catch(err => console.error("Security audit fetch error:", err));
    }


    // --- AUTONOMOUS ROBOTICS & EKF NAVIGATION ---
    function fetchRoboticsStep() {
        fetch('/api/robotics/step')
            .then(res => res.json())
            .then(data => {
                document.getElementById('ekf-error-val').textContent = data.ekf_error_m;
                document.getElementById('gps-error-val').textContent = data.gps_jammed ? 'Sinyal Yok (EW)' : data.raw_gps_error_m;
                document.getElementById('ekf-uncertainty-val').textContent = data.uncertainty_m;
                
                document.getElementById('robot-true-pos').textContent = `X: ${data.true_position.x}, Y: ${data.true_position.y}`;
                document.getElementById('robot-ekf-pos').textContent = `X: ${data.ekf_estimated.x}, Y: ${data.ekf_estimated.y}`;
                
                const statusEl = document.getElementById('robot-gps-status');
                if (data.gps_jammed) {
                    statusEl.textContent = 'GPS Kesintisi (Ölü Seyir - INS Modu)';
                    statusEl.className = 'text-amber';
                } else {
                    statusEl.textContent = 'Aktif (Sinyal Güçlü & EKF Güncelleniyor)';
                    statusEl.className = 'text-cyan';
                }
            })
            .catch(err => console.error("Robotics step error:", err));
    }

    const toggleRoboticsGpsBtn = document.getElementById('toggle-robotics-gps-btn');
    if (toggleRoboticsGpsBtn) {
        toggleRoboticsGpsBtn.addEventListener('click', () => {
            fetch('/api/robotics/toggle-gps-jamming', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.gps_jammed) {
                        toggleRoboticsGpsBtn.classList.add('active');
                        toggleRoboticsGpsBtn.innerHTML = '<i class="fa-solid fa-satellite-dish"></i> GPS Kesintisini Kaldır';
                    } else {
                        toggleRoboticsGpsBtn.classList.remove('active');
                        toggleRoboticsGpsBtn.innerHTML = '<i class="fa-solid fa-satellite-dish"></i> GPS Kesintisi (EW Jamming) Oluştur';
                    }
                    fetchRoboticsStep();
                })
                .catch(err => console.error("Toggle GPS error:", err));
        });
    }

    const runPqcTunnelBtn = document.getElementById('run-pqc-tunnel-btn');
    const pqcConsoleDisplay = document.getElementById('pqc-console-display');
    if (runPqcTunnelBtn) {
        runPqcTunnelBtn.addEventListener('click', () => {
            fetch('/api/pqc/test-tunnel', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    const pkt = data.packet;
                    let log = `<strong>PQC Kyber/Dilithium Tünel Testi:</strong> <span class="text-emerald">BAŞARILI</span><br><br>`;
                    log += `<strong>Gönderici:</strong> ${pkt.sender} ➔ <strong>Alıcı:</strong> ${pkt.receiver}<br>`;
                    log += `<strong>KEM:</strong> ${pkt.kem} | <strong>İmza:</strong> ${pkt.sig}<br>`;
                    log += `<strong>Şifreli Paket:</strong> <span class="text-cyan">${pkt.ciphertext.substring(0, 32)}...</span><br>`;
                    log += `<strong>Doğrulanan Özgün Veri:</strong> ${JSON.stringify(data.decrypted_payload)}`;
                    
                    pqcConsoleDisplay.innerHTML = `<div class="chat-bubble bot"><p>${log}</p></div>`;
                })
                .catch(err => console.error("PQC Tunnel test error:", err));
        });
    }

});
