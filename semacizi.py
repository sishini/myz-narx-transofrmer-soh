import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_final_corrected_architecture():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 15)
    ax.axis('off')

    # Renk Paleti
    color_in = '#E1F5FE'; color_back = '#BBDEFB'; color_emb = '#90CAF9'
    color_neck = '#C8E6C9'; color_narx = '#FFF9C4'; color_dec = '#A5D6A7'
    color_head = '#FFCDD2'; color_out = '#EF9A9A'

    # 1. GİRDİ (Input)
    ax.add_patch(patches.FancyBboxPatch((3.5, 13.5), 3, 0.8, boxstyle="round,pad=0.1", ec="black", fc=color_in))
    ax.text(5, 13.9, "GİRDİ (Input)\n(32, 7, 256, 4)", ha='center', va='center', fontsize=9, weight='bold')

    # 2. BACKBONE (Cluster)
    ax.add_patch(patches.Rectangle((2, 10.5), 6, 2.5, linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--'))
    ax.text(5, 12.6, "1. BACKBONE (Özellik Çıkarımı)", ha='center', fontsize=10, weight='bold', color='blue')
    ax.add_patch(patches.FancyBboxPatch((3.5, 11.5), 3, 0.5, boxstyle="round,pad=0.1", ec="black", fc=color_back))
    ax.text(5, 11.75, "Conv2D (16 filtre, 3x3)", ha='center', va='center', fontsize=8)
    ax.add_patch(patches.FancyBboxPatch((3.5, 10.8), 3, 0.5, boxstyle="round,pad=0.1", ec="black", fc=color_back))
    ax.text(5, 11.05, "Conv2D (16 filtre, 3x3)", ha='center', va='center', fontsize=8)

    # 3. EMBEDDED (Ara Çıktı - Dışarıda)
    ax.add_patch(patches.FancyBboxPatch((3.5, 9.0), 3, 0.7, boxstyle="round,pad=0.1", ec="black", fc=color_emb))
    ax.text(5, 9.35, "Embedded Vectors\n(32, 7, 16)", ha='center', va='center', fontsize=9, weight='bold')

    # 4. NECK (Cluster)
    ax.add_patch(patches.Rectangle((1.5, 4.2), 7, 4.2, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--'))
    ax.text(5, 8.0, "2. NECK (Fusion Zone)", ha='center', fontsize=10, weight='bold', color='green')
    
    # Encoder (Sol)
    ax.add_patch(patches.FancyBboxPatch((2, 6.5), 2.5, 0.8, boxstyle="round,pad=0.1", ec="black", fc=color_neck))
    ax.text(3.25, 6.9, "Transformer\nEncoder (16)", ha='center', va='center', fontsize=8)
    
    # NARX (Sağ) - [16, 6]
    ax.add_patch(patches.FancyBboxPatch((5.5, 6.5), 2.5, 0.8, boxstyle="round,pad=0.1", ec="black", fc=color_narx))
    ax.text(6.75, 6.9, "NARX (Linear)\n(6 -> 16)", ha='center', va='center', fontsize=8)
    
    # Decoder (Alt)
    ax.add_patch(patches.FancyBboxPatch((3, 4.6), 4, 0.8, boxstyle="round,pad=0.1", ec="black", fc=color_dec))
    ax.text(5, 5.0, "Transformer Decoder\n(Cross-Attention)", ha='center', va='center', fontsize=8)

    # 5. HEAD (Cluster)
    ax.add_patch(patches.Rectangle((3, 1.8), 4, 1.8, linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--'))
    ax.text(5, 3.3, "3. HEAD (Regresyon)", ha='center', fontsize=10, weight='bold', color='red')
    
    # Final Linear [1, 16]
    ax.add_patch(patches.FancyBboxPatch((3.5, 2.2), 3, 0.8, boxstyle="round,pad=0.1", ec="black", fc=color_head))
    ax.text(5, 2.6, "Final Linear Layer\n(16 -> 1)", ha='center', va='center', fontsize=8)

    # 6. TAHMİN (Nihai Çıktı - Dışarıda)
    ax.add_patch(patches.FancyBboxPatch((3.5, 0.2), 3, 0.8, boxstyle="round,pad=0.1", ec="black", fc=color_out))
    ax.text(5, 0.6, "TAHMİNİ SoH\n(16, 1)", ha='center', va='center', fontsize=9, weight='bold')

    # OKLAR (Source -> Target)
    kw = dict(arrowstyle='->', lw=1.2, color='black')
    ax.annotate('', xy=(5, 12.2), xytext=(5, 13.5), arrowprops=kw) # In to Backbone
    ax.annotate('', xy=(5, 9.7), xytext=(5, 10.5), arrowprops=kw)  # Backbone to Emb
    ax.annotate('', xy=(3.25, 7.3), xytext=(5, 9.0), arrowprops=kw) # Emb to Encoder
    ax.annotate('', xy=(6.75, 7.3), xytext=(5, 9.0), arrowprops=kw) # Emb to NARX
    ax.annotate('', xy=(4.5, 5.4), xytext=(3.25, 6.5), arrowprops=kw) # Encoder to Decoder
    ax.annotate('', xy=(5.5, 5.4), xytext=(6.75, 6.5), arrowprops=kw) # NARX to Decoder
    ax.annotate('', xy=(5, 3.6), xytext=(5, 4.6), arrowprops=kw)   # Decoder to Head
    ax.annotate('', xy=(5, 1.0), xytext=(5, 2.2), arrowprops=kw)   # Head to Out

    plt.title("NARX-Transformer Mimari Akış Diyagramı", fontsize=12, weight='bold', pad=20)
    plt.show()

draw_final_corrected_architecture()