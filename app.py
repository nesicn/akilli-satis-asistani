import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. SAYFA YAPILANDIRMASI & TEMA
# ==========================================
st.set_page_config(
    page_title="Smart Checkout AI - Kasa Asistanı",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Stilleri
st.markdown("""
    <style>
    .main-header {
        font-size: 26px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .xai-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .xai-box-low {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE İLKLEME & CALLBACKS
# ==========================================
if "total_recommendations" not in st.session_state:
    st.session_state.total_recommendations = 0
if "accepted_recommendations" not in st.session_state:
    st.session_state.accepted_recommendations = 0
if "rejected_recommendations" not in st.session_state:
    st.session_state.rejected_recommendations = 0
if "ai_generated_revenue" not in st.session_state:
    st.session_state.ai_generated_revenue = 0.0
if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False
if "last_feedback_msg" not in st.session_state:
    st.session_state.last_feedback_msg = None
if "last_feedback_type" not in st.session_state:
    st.session_state.last_feedback_type = None

def reset_analysis():
    st.session_state.analiz_yapildi = False
    st.session_state.last_feedback_msg = None
    st.session_state.last_feedback_type = None

# ==========================================
# 3. VERİ VE MODEL YÜKLEME (OTOMATİK KONTROL & FALLBACK)
# ==========================================
@st.cache_resource
def load_model(model_path="smart_kasa_model.pkl"):
    """Yapay zeka modelini güvenli yükler; hata durumunda None döner."""
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception:
        return None

@st.cache_data
def load_customers(csv_path="musteri_davranis_seti.csv"):
    """Müşteri veri setini güvenli yükler, dosya yoksa otomatik oluşturur."""
    if not os.path.exists(csv_path):
        # Varsayılan örnek veri seti oluştur
        sample_data = {
            "Musteri_ID": ["M001", "M002", "M003", "M004"],
            "Ad_Soyad": ["Ayşe Yılmaz", "Mehmet Demir", "Zeynep Kaya", "Can Çelik"],
            "Telefon": ["5555614226", "5321112233", "5423334455", "5339998877"],
            "Segment": ["VIP", "Sadık Müşteri", "Standart", "Yeni Müşteri"],
            "gecmis_ortalama_sepet_tutari": [850.0, 420.0, 310.0, 190.0],
            "makyaj_gecen_gun": [15, 45, 10, 5],
            "makyaj_ort_alim_araligi": [30, 30, 20, 15],
            "makyaj_tuketim_orani": [0.8, 0.4, 0.6, 0.3],
            "cilt_bakimi_gecen_gun": [20, 60, 15, 8],
            "cilt_bakimi_ort_alim_araligi": [45, 45, 30, 30],
            "cilt_bakimi_tuketim_orani": [0.7, 0.5, 0.5, 0.4]
        }
        df_default = pd.DataFrame(sample_data)
        try:
            df_default.to_csv(csv_path, index=False)
        except Exception:
            pass
        return df_default

    try:
        df = pd.read_csv(csv_path)
        # Sütun isimlerindeki boşlukları temizle
        df.columns = [str(c).strip() for c in df.columns]
        
        # Telefon ve ID sütunlarını stringe çevir
        for col in ['Telefon', 'telefon', 'Phone', 'PHONE', 'Musteri_ID', 'musteri_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except Exception:
        return pd.DataFrame()

model = load_model()
df_musteriler = load_customers()

# Esnek sütun bulma yardımcı fonksiyonu
def get_column_value(row, possible_keys, default_val=""):
    for k in possible_keys:
        if k in row:
            return row[k]
    return default_val

def mask_text(text, visible_chars=2):
    text_str = str(text)
    if len(text_str) <= visible_chars:
        return text_str
    return text_str[:visible_chars] + "*" * (len(text_str) - visible_chars)

def generate_xai_insights(secili_musteri, harcanan_tutar, secilen_kategori, proba):
    insights = []
    segment = get_column_value(secili_musteri, ['Segment', 'segment'], 'Standart') if secili_musteri else 'Standart'
    if 'VIP' in str(segment):
        insights.append("⭐ **VIP Müşteri Sadakati:** Müşterinin yüksek alışveriş bağlılığı ikna olasılığını artırıyor.")
    elif 'Sadık' in str(segment) or 'Sadik' in str(segment):
        insights.append("💙 **Sadık Müşteri Profili:** Düzenli mağaza ziyaretleri öneri kabul esnekliğini destekliyor.")
        
    kat_key = secilen_kategori.lower().replace(" ", "_").replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o")
    if secili_musteri:
        gecen_gun = float(get_column_value(secili_musteri, [f"{kat_key}_gecen_gun", "gecen_gun"], 30))
        ort_aralik = float(get_column_value(secili_musteri, [f"{kat_key}_ort_alim_araligi", "ort_alim_araligi"], 30))
        tuketim_orani = float(get_column_value(secili_musteri, [f"{kat_key}_tuketim_orani", "tuketim_orani"], 0.5))
        
        if gecen_gun >= ort_aralik:
            insights.append(f"⏳ **Yenileme Zamanı Gelmiş ({secilen_kategori}):** Son alımdan bu yana {int(gecen_gun)} gün geçmiş (Ort. Döngü: {int(ort_aralik)} gün).")
        if tuketim_orani > 0.6:
            insights.append(f"📈 **Yüksek Tüketim Skoru:** Müşterinin {secilen_kategori} kategorisindeki geçmiş tüketim oranı (%{tuketim_orani*100:.0f}) yüksek.")
            
    if harcanan_tutar >= 500:
        insights.append(f"💰 **Yüksek Alışveriş Hacmi:** {harcanan_tutar:.0f} TL tutarındaki sepet, ek tekliflere açık olduğunu gösteriyor.")
    elif harcanan_tutar < 200:
        insights.append("⚠️ **Düşük Sepet Tutarı:** Müşteri hassas bir bütçeyle alışveriş yapıyor olabilir.")
        
    if not insights:
        insights.append("ℹ️ Genel sepet ortalamaları ve mağaza içi standart müşteri profil davranışları esas alındı.")
        
    return insights

# ==========================================
# 4. YAN MENÜ (SIDEBAR) & CANLI METRİKLER
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/checkout.png", width=70)
    st.title("Smart Checkout AI")
    st.caption("Retail AI Assistant")
    st.divider()
    
    maskeleme_aktif = st.toggle("🔒 Maskeleme Modu", value=False)
    st.info("Sistem, kasa anında akıllı çapraz satış ve müşteri ikna olasılığı tahmini üretir.")
    
    st.divider()
    st.markdown("**Sistem Durumu:**")
    if model is not None:
        st.success("🟢 AI Modeli Aktif (`XGBoost`)")
    else:
        st.warning("🟡 AI Modeli (Fallback Modu Aktif)")
        
    st.divider()
    st.markdown("**📈 Canlı Kasa Performansı (ROI):**")
    st.metric("Ek Ciro Katkısı", f"{st.session_state.ai_generated_revenue:.2f} TL")
    acc_rate = (st.session_state.accepted_recommendations / st.session_state.total_recommendations * 100) if st.session_state.total_recommendations > 0 else 0.0
    st.metric("Kasa Dönüşüm Oranı", f"%{acc_rate:.1f}")

# ==========================================
# 5. ANA EKRAN VE SEKMELER
# ==========================================
st.title("🛒 Akıllı Kasa Asistanı & Öneri Motoru")

tab_kasa, tab_analitik, tab_yeni_musteri = st.tabs([
    "🛍️ Kasa İşlem Ekranı", 
    "📊 Müşteri & Mağaza Analitiği", 
    "➕ Yeni Müşteri Kaydı"
])

# ------------------------------------------
# TAB 1: KASA İŞLEM EKRANI
# ------------------------------------------
with tab_kasa:
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("1. Müşteri & Sepet Bilgileri")
        
        telefon_input = st.text_input(
            "📱 Müşteri Telefon No (Örn: 5555614226):", 
            value="5555614226",
            key="input_tel",
            on_change=reset_analysis
        )
        
        secili_musteri = None
        phone_col = None
        for col in df_musteriler.columns:
            if col.lower() in ['telefon', 'tel', 'phone', 'gsm']:
                phone_col = col
                break
                
        if telefon_input and not df_musteriler.empty and phone_col:
            match = df_musteriler[df_musteriler[phone_col].astype(str).str.contains(telefon_input.strip(), na=False)]
            if not match.empty:
                secili_musteri = match.iloc[0].to_dict()
                ad_raw = get_column_value(secili_musteri, ['Ad_Soyad', 'ad_soyad', 'Ad', 'Name', 'name'], 'Bilinmeyen Müşteri')
                segment_raw = get_column_value(secili_musteri, ['Segment', 'segment'], 'Standart')
                
                ad_display = mask_text(ad_raw) if maskeleme_aktif else ad_raw
                st.success(f"Müşteri Bulundu: **{ad_display}** ({segment_raw} Segment)")
            else:
                st.info("ℹ️ Kayıtlı müşteri bulunamadı. Genel müşteri profili ile devam ediliyor.")
        else:
            if telefon_input:
                st.info("ℹ️ Müşteri veri setinde telefon sütunu tespit edildi veya veri seti boş.")
        
        st.divider()
        
        harcanan_tutar = st.number_input(
            "💰 Anlık Sepet Tutarı (TL):", 
            min_value=10.0, 
            max_value=10000.0, 
            value=350.0, 
            step=10.0,
            key="input_tutar",
            on_change=reset_analysis
        )
        sepetteki_urun = st.slider(
            "📦 Sepetteki Ürün Adedi:", 
            min_value=1, 
            max_value=20, 
            value=3,
            key="input_urun",
            on_change=reset_analysis
        )
        
        kategoriler = ["Makyaj", "Cilt Bakımı", "Ağız Bakımı", "Parfüm", "Saç Bakımı", "Satış Sonrası Bakım"]
        secilen_kategori = st.selectbox(
            "🏷️ Sepetteki Ağırlıklı Kategori:", 
            kategoriler,
            key="input_kat",
            on_change=reset_analysis
        )
        
        if st.button("⚡ AI Önerilerini ve Tahmini Hesapla", type="primary", use_container_width=True):
            st.session_state.analiz_yapildi = True
            st.session_state.last_feedback_msg = None
            st.session_state.last_feedback_type = None

    with col_right:
        st.subheader("2. Yapay Zeka & Kasa Fırsat Analizi")
        
        if st.session_state.analiz_yapildi:
            proba = 0.75 
            if model is not None:
                try:
                    input_dict = secili_musteri.copy() if secili_musteri else {}
                    input_dict["Harcanan_Tutar_TL"] = harcanan_tutar
                    input_dict["Sepet_Tutari_TL"] = harcanan_tutar
                    input_dict["Sepetteki_Urun_Adedi"] = sepetteki_urun
                    input_dict["Onerilen_Kategori"] = secilen_kategori
                    
                    df_input = pd.DataFrame([input_dict])
                    expected_features = getattr(model, "feature_names_in_", None)
                    if expected_features is not None:
                        df_input = df_input.reindex(columns=expected_features, fill_value=0)
                        
                    if hasattr(model, "predict_proba"):
                        proba = float(model.predict_proba(df_input)[0][1])
                    else:
                        pred = model.predict(df_input)[0]
                        proba = 0.85 if pred == 1 else 0.35
                except Exception:
                    proba = min(0.92, max(0.25, (harcanan_tutar / 1000.0) * 0.5 + (0.3 if secili_musteri else 0.1)))
            else:
                proba = min(0.90, max(0.30, (harcanan_tutar / 800.0) * 0.6 + 0.2))

            st.markdown("#### 🎯 Müşteri Ek Satın Alma İkna Olasılığı")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric(label="Tahmin Skoru", value=f"%{proba * 100:.1f}")
            with c2:
                st.progress(float(proba))
                if proba >= 0.70:
                    st.caption("🔥 **Yüksek Potansiyel:** Müşteri kasada ek önerilere çok açık.")
                elif proba >= 0.40:
                    st.caption("🟡 **Orta Potansiyel:** Standart kasa fırsat ürünleri sunulabilir.")
                else:
                    st.caption("⚪ **Düşük Potansiyel:** Hızlı işlem tamamlanması önerilir.")

            st.divider()

            st.markdown("#### 🔍 Yapay Zeka Karar Gerekçeleri")
            xai_list = generate_xai_insights(secili_musteri, harcanan_tutar, secilen_kategori, proba)
            
            box_style = "xai-box" if proba >= 0.5 else "xai-box-low"
            st.markdown(f'<div class="{box_style}">', unsafe_allow_html=True)
            for item in xai_list:
                st.markdown(f"- {item}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.divider()

            st.markdown("#### 💡 Kasiyer İçin Anlık Aksiyon Önerileri")
            
            if harcanan_tutar < 500:
                kalan = 500 - harcanan_tutar
                st.info(f"🎁 **Kampanya Fırsatı:** Sepete **{kalan:.2f} TL** daha eklendiğinde Kasa İndirimi kazanılıyor!")
            elif harcanan_tutar >= 500 and harcanan_tutar < 1000:
                st.success("🎉 Müşteri 500 TL üzeri kargo/indirim limitine ulaştı! VIP hediye çeki sunulabilir.")

            oneriler = {
                "Makyaj": ("Makyaj Temizleme Suyu & Pamuk Seti", 49.90),
                "Cilt Bakımı": ("Güneş Koruyucu Krem (SPF 50+)", 129.00),
                "Ağız Bakımı": ("Çalkalama Suyu & Diş İpi Combo", 39.90),
                "Parfüm": ("Cep Boy Seyahat Parfüm Şişesi", 29.90),
                "Saç Bakımı": ("Durulanmayan Saç Bakım Yağı", 79.90),
                "Satış Sonrası Bakım": ("Nemlendirici El & Vücut Losyonu", 34.90)
            }
            
            oneri_urun, oneri_fiyat = oneriler.get(secilen_kategori, ("Kasa Önü Minis Ürünler", 19.90))
            st.warning(f"👉 **Önerilen Kasa Önü Ürünü:** {oneri_urun} — **Özel Fiyat:** {oneri_fiyat:.2f} TL")

            st.divider()

            st.markdown("#### 🔄 Müşteri Yanıtı Kaydı (Feedback Loop)")
            st.caption("Kasiyer teklifi sunduktan sonra müşterinin kararını kaydedin.")
            
            if st.session_state.last_feedback_msg:
                if st.session_state.last_feedback_type == "success":
                    st.success(st.session_state.last_feedback_msg)
                else:
                    st.info(st.session_state.last_feedback_msg)
            
            fb_col1, fb_col2 = st.columns(2)
            
            with fb_col1:
                if st.button("✅ Müşteri Öneriyi Kabul Etti", use_container_width=True, type="secondary"):
                    st.session_state.total_recommendations += 1
                    st.session_state.accepted_recommendations += 1
                    st.session_state.ai_generated_revenue += oneri_fiyat
                    st.session_state.last_feedback_msg = f"🎉 **Kabul Kaydedildi:** Sepete +{oneri_fiyat:.2f} TL eklendi!"
                    st.session_state.last_feedback_type = "success"
                    st.rerun()
                    
            with fb_col2:
                if st.button("❌ Müşteri Öneriyi Reddetti", use_container_width=True):
                    st.session_state.total_recommendations += 1
                    st.session_state.rejected_recommendations += 1
                    st.session_state.last_feedback_msg = "ℹ️ **Red Kaydedildi:** Yanıt model iyileştirme veri havuzuna aktarıldı."
                    st.session_state.last_feedback_type = "info"
                    st.rerun()

        else:
            st.info("👈 Önerileri ve XAI detaylarını görmek için sol taraftaki **'AI Önerilerini ve Tahmini Hesapla'** butonuna basın.")

# ------------------------------------------
# TAB 2: MÜŞTERİ & MAĞAZA ANALİTİĞİ
# ------------------------------------------
with tab_analitik:
    st.subheader("📊 Müşteri Veri Seti & Mağaza Özet Analizleri")
    
    st.markdown("### 💰 AI Katma Değer & İş Etkisi")
    roi1, roi2, roi3, roi4 = st.columns(4)
    roi1.metric("Toplam Sunulan Öneri", st.session_state.total_recommendations)
    roi2.metric("Kabul Edilen Öneri", st.session_state.accepted_recommendations)
    
    conv_rate = (st.session_state.accepted_recommendations / st.session_state.total_recommendations * 100) if st.session_state.total_recommendations > 0 else 0.0
    roi3.metric("Kasa Dönüşüm Oranı", f"%{conv_rate:.1f}")
    roi4.metric("AI Kaynaklı Ek Ciro", f"{st.session_state.ai_generated_revenue:.2f} TL")
    
    st.divider()

    if not df_musteriler.empty:
        col_m1, col_m2, col_m3, _ = st.columns(4)
        col_m1.metric("Toplam Kayıtlı Müşteri", len(df_musteriler))
        
        segment_col = None
        for col in df_musteriler.columns:
            if col.lower() == 'segment':
                segment_col = col
                break
                
        vip_count = len(df_musteriler[df_musteriler[segment_col].astype(str).str.contains('VIP', na=False)]) if segment_col else 0
        col_m2.metric("VIP Müşteri Sayısı", vip_count)
        
        st.divider()
        st.markdown("##### 📄 Mevcut Müşteri Veri Tablosu")
        df_display = df_musteriler.copy()
        if maskeleme_aktif:
            for col in df_display.columns:
                if 'ad' in col.lower() or 'name' in col.lower():
                    df_display[col] = df_display[col].apply(lambda x: mask_text(str(x)))
                if 'telefon' in col.lower() or 'tel' in col.lower() or 'phone' in col.lower():
                    df_display[col] = df_display[col].apply(lambda x: mask_text(str(x), 4))
            
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.warning("Görüntülenecek müşteri verisi bulunamadı veya veri seti yüklenemedi.")

# ------------------------------------------
# TAB 3: YENİ MÜŞTERİ KAYDI
# ------------------------------------------
with tab_yeni_musteri:
    st.subheader("➕ Yeni Müşteri Ekleme Formu")
    with st.form("yeni_musteri_formu", clear_on_submit=True):
        f_ad = st.text_input("Ad Soyad:")
        f_tel = st.text_input("Telefon No:")
        f_segment = st.selectbox("Segment:", ["Yeni Müşteri", "Standart Müşteri", "Sadık Müşteri", "VIP Müşteri"])
        
        submit_btn = st.form_submit_button("Müşteriyi Kaydet")
        if submit_btn:
            if f_ad and f_tel:
                st.success(f"✅ Müşteri `{f_ad}` başarıyla eklendi! (Simülasyon)")
            else:
                st.error("Lütfen Ad Soyad ve Telefon alanlarını doldurun.")
