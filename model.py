import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

world_size = 1
rank = 0
block_size = 64 


@dataclass
class ModelArgs:
    """
Model argümanlarını ve hiperparametreleri tanımlayan veri sınıfı.

Öznitelikler (Attributes):
    max_batch_size (int): Maksimum batch (yığın) boyutu.
    max_seq_len (int): Maksimum dizi (sequence) uzunluğu.
    dtype (Literal["bf16", "fp8"]): Hesaplamalar için kullanılacak veri tipi.
    vocab_size (int): Kelime dağarcığı (vocabulary) boyutu.
    dim (int): Modelin genel gizli katman boyutu (embedding + hidden dim).
    inter_dim (int): MLP (besleyici ağ) katmanları için ara katman boyutu.
    moe_inter_dim (int): MoE (Mixture of Experts) katmanları için ara katman boyutu.
    n_layers (int): Transformer katmanı sayısı.
    n_dense_layers (int): Modeldeki yoğun (dense) katman sayısı.
    n_heads (int): Dikkat (attention) başlığı sayısı.
    n_routed_experts (int): MoE içinde yönlendirilen uzman sayısı.
    n_shared_experts (int): MoE içinde paylaşılan (her gruba açık) uzman sayısı.
    n_activated_experts (int): Her örnek için aktif edilen uzman sayısı.
    n_expert_groups (int): Uzman grubu sayısı (MoE routing grupları).
    n_limited_groups (int): MoE yönlendirmesinde sınırlandırılmış grup sayısı.
    score_func (Literal["softmax", "sigmoid"]): MoE yönlendirme puanlama fonksiyonu.
    route_scale (float): Routing skorları için çarpan ölçekleme katsayısı.
    q_lora_rank (int): Query (sorgu) projeksiyonları için LoRA rank’ı.
    kv_lora_rank (int): Key-Value (anahtar-değer) projeksiyonları için LoRA rank’ı.
    qk_nope_head_dim (int): Konumsal bilgi olmadan QK projeksiyonları için başlık boyutu.
    qk_rope_head_dim (int): Rotary Positional Embedding kullanılan QK projeksiyon başlık boyutu.
    v_head_dim (int): Value (değer) projeksiyon başlık boyutu.
    original_seq_len (int): Modelin önceden eğitim aldığı maksimum dizgi uzunluğu.
    rope_theta (float): Rotary positional encoding için temel (üstel frekans) değeri.
    rope_factor (float): Rotary frekans düzeltmesi için ölçekleme katsayısı.
    beta_fast (int): Düşük rotasyon eşiği (erken düzeltme için).
    beta_slow (int): Yüksek rotasyon eşiği (tam düzeltme için).
    mscale (float): Uzatılmış dikkat (extended attention) için ölçekleme katsayısı.
    """
    max_batch_size: int = 2
    max_seq_len: int = 256
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 12564
    dim: int = 1024
    inter_dim: int = 4 * dim
    moe_inter_dim: int = 704
    n_layers: int = 6
    n_dense_layers: int = 1
    n_heads: int = 8
    # moe
    n_routed_experts: int = 4
    n_shared_experts: int = 2
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    # yarn
    original_seq_len: int = 512
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

    # data preparing
    shuffle: bool = True
    drop_last: bool = True

    # training
    train:bool = True
    dataset_path = "/kaggle/working/c4_tr-1m.txt" if os.path.exists("/kaggle/working/c4_tr-1m.txt") else "8k_data.txt"

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
Rotary pozisyonel gömmeler (rotary positional embeddings) için frekansa dayalı kompleks üstel değerleri önceden hesaplar.

Parametreler (Args):
    args (ModelArgs): Pozisyonel gömme parametrelerini içeren model argümanları.

Dönüş (Returns):
    torch.Tensor: Pozisyonlara karşılık gelen karmaşık (complex) üstel değerleri içeren bir tensor.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast # frekans limits
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    #? Belirtilen rotasyon sayısı için dönme açısı 2π·num_rot eşik değerini geçen boyut indeksini hesaplar
    def find_correction_dim(num_rotations, dim, base, max_seq_len):

        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    #? Dönme açısının bozulmaya başladığı ve tamamen bozulduğu boyut aralığını belirler
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):

        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    #? Belirtilen aralıkta [0,1] arasında doğrusal artan bir geçiş (ramp) vektörü oluşturur
    def linear_ramp_factor(min, max, dim):

        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    #? Eğer dizi uzunluğu pretraining sınırını aşıyorsa, frekansları yumuşakça düzelt
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:

    assert x.shape[-1] % 2 == 0, "Rotary dim must be divisible by 2!"
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).reshape(*x.shape[:-1], -1)
    return y.to(dtype)

class RMSNorm(nn.Module):

    def __init__(self, dim:int, eps:float=1e-3):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+ self.eps)

    def forward(self, x:torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class MLA(nn.Module):

    """
        Öznitelikler (Attributes):
            dim (int): Girdi özelliklerinin boyutu (modelin genel gizli boyutu).
            n_heads (int): Dikkat (attention) başlığı sayısı.
            n_local_heads (int): Dağıtık sistemler için kullanılan lokal attention başlığı sayısı.
            q_lora_rank (int): Query projeksiyonları için düşük-rank (low-rank) LoRA matrislerinin rank değeri.
            kv_lora_rank (int): Key/Value projeksiyonları (C^kv) için düşük-rank LoRA rank değeri.
            qk_nope_head_dim (int): Konumsal bilgi içermeyen query/key projeksiyonlarının boyutu.
            qk_rope_head_dim (int): Rotary positional encoding uygulanan query/key projeksiyonlarının boyutu.
            qk_head_dim (int): Query ve key projeksiyonlarının toplam boyutu.
            v_head_dim (int): Value (değer) projeksiyonlarının boyutu.
            softmax_scale (float): Attention hesaplamalarında softmax’a uygulanan ölçekleme faktörü.
    """
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_head = args.n_heads
        self.n_local_head = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.isTrain = args.train

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_head * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank) # W_DQ
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_head * self.qk_head_dim) # in features: c_t^Q  out features: q_t^C
        
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim) # burada W^DKV ile W_ht^Kr hesaplamaları birliştirildi
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_head * (self.qk_nope_head_dim + self.v_head_dim)) # burada W^uk  x c_t^kv işlemi ile W^uv x c_t^kv işlemleri birleştiriliyor
        self.wo = nn.Linear(self.n_head * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale


        self.register_buffer('kv_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False) # K/V head'lerinin üretildi latent space
        self.register_buffer('pe_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False) # Pozisyon bilgisini bellekte tutma

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]):
        batch_size, seq_len, _ = x.size()
        end_pos = start_pos + seq_len
        
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x))) # full q_t^c query vector

        q = q.view(batch_size,seq_len, self.n_local_head, self.qk_head_dim) # Divide q into heads
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # birlikte hesapladığımız q ve q_rope değerlerini ayırıyoruz
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) # k_pe batch_size içermediğinden ona batch_size boyutu ekliyoruz
        

        # deepseek tarzı attention hesaplaması
        wkv_b = self.wkv_b.weight
        wkv_b = wkv_b.view(self.n_local_head, -1, self.kv_lora_rank)
        q_nope = torch.einsum('bshd,hdc->bshc', q_nope, wkv_b[:, :self.qk_nope_head_dim])
        if not self.isTrain:
                    
            self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:batch_size, start_pos:end_pos] = k_pe.squeeze(2)

        assert q_nope.shape[-1] == self.kv_cache.shape[-1], "Head dim mismatch between q_nope and kv_cache" 
        kv = self.kv_cache[:batch_size, :end_pos].unsqueeze(2)  # -> [B, T, 1, R]
        pe = self.pe_cache[:batch_size, :end_pos].unsqueeze(2)  # -> [B, T, 1, R]
        scores = (
             torch.einsum('bshr,bthr->bsht', q_nope, kv) +
             torch.einsum('bshr,bthr->bsht', q_pe, pe)
            ) * self.softmax_scale

        if mask is None and end_pos > 1:
            mask = torch.full((end_pos, end_pos), float('-inf'), device=x.device).triu(1)

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        x = torch.einsum('bsht,btc->bshc',scores, self.kv_cache[:batch_size, :end_pos])
        x = torch.einsum('bshc,hdc->bshd',x,wkv_b[:,-self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x

class Gate(nn.Module):
    """
        Bir mixture-of-experts (MoE) modelinde girdileri uzmanlara yönlendirmek için kullanılan gating (yönlendirme) mekanizması.

        Öznitelikler (Attributes):
            dim (int): Girdi özelliklerinin boyutu.
            topk (int): Her giriş için aktif edilecek en iyi (en yüksek puanlı) uzman sayısı.
            n_groups (int): Routing işlemi için kullanılan toplam grup sayısı.
            topk_groups (int): Girdilerin yönlendirileceği en yüksek puanlı grup sayısı.
            score_func (str): Yönlendirme skoru hesaplamada kullanılan aktivasyon fonksiyonu ('softmax' veya 'sigmoid').
            route_scale (float): Routing skorlarını ölçeklendirmek için kullanılan çarpan katsayısı.
            weight (torch.nn.Parameter): Gating mekanizması için öğrenilebilir ağırlık parametresi.
            bias (Optional[torch.nn.Parameter]): Opsiyonel olarak kullanılan bias (sapma) terimi.
    """

    def __init__(self, args:ModelArgs): 
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.topk_groups = args.n_limited_groups
        self.n_groups = args.n_expert_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts))

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Routing (yönlendirme) ağırlıkları (skorlar), 
                - Seçilen uzmanların indeksleri.
        """
        # Skorları hesapla: her token tüm uzmanlara karşı skor üretir
        scores = F.linear(x, self.weight, self.bias)
        if self.score_func == 'softmax':
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        original_scores = scores
        #? Sık seçilen uzmanlara ait skorları azaltmak (veya az seçilenleri yükseltmek) için kullanılır.
        if self.bias is not None:
            scores = scores + self.bias
        
        #? Eğer uzmanlar gruplara ayrılmışsa, grup seviyesinde top-k gruplar seçilir
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2,dim=-1)[0].sum(dim=-1)
            # En yüksek skorlu grupları seç
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # Geri kalan grupları maskele (uzmanlar çalışmasın)
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float('-inf')).flatten(1)
        
        # Uzmanlar arasında top-k seçimi (örneğin 2 uzman aktif edilecekse)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # Orijinal skor matrisinden seçilen uzmanların skorlarını al
        weights = original_scores.gather(1, indices)

        if self.score_func == 'sigmoid':
            weights = weights /( weights.sum(dim=-1, keepdim=True))
        
        weights =weights* self.route_scale

        self.last_scores = original_scores.detach()  # [B*T, N_r]
        self.last_topk = indices.detach()  
        return weights.type_as(x), indices

class MLP(nn.Module):
    """
        İleri beslemeli katman (feed-forward layer) olarak kullanılan Çok Katmanlı Algılayıcı (Multi-Layer Perceptron, MLP).

        Öznitelikler (Attributes):
            w1 : Girdi → gizli katman dönüşümü için lineer katman.
            w2 : Gizli katman → çıktı dönüşümü için lineer katman.
            w3 : Özellik dönüşümü için ek bir lineer katman.
    """
    def __init__(self, dim:int, inter_dim:int):

        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x:torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Expert(nn.Module):

    def __init__(self, dim:int, inter_dim:int):

        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x:torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    """
        Uzmanlardan oluşan karışım (Mixture-of-Experts, MoE) modülü.

        Öznitelikler (Attributes):
            dim (int): Girdi özelliklerinin boyutu.
            n_routed_experts (int): Modelde yer alan toplam uzman (expert) sayısı.
            n_local_experts (int): Dağıtık sistemlerde yerel olarak yönetilen uzman sayısı.
            n_activated_experts (int): Her bir giriş için aktif edilen uzman sayısı.
            gate (nn.Module): Girdileri uygun uzmanlara yönlendirmek için kullanılan gating (yönlendirme) mekanizması.
            experts (nn.ModuleList): Uzman modüllerinin (ağlarının) listesi.
            shared_experts (nn.Module): Tüm girdilere uygulanan paylaşımlı uzmanlar.
    """
    def __init__(self, args: ModelArgs):

        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        # distributed MoE için
        self.n_local_experts = args.n_routed_experts // world_size
        self.expert_start_idx = rank * self.n_local_experts
        self.expert_end_idx = self.expert_start_idx + self.n_local_experts
        self.gate = Gate(args)
        
        # routed expertler
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.expert_start_idx <= i < self.expert_end_idx else None for i in range(self.n_routed_experts)])
        # her inputun geçeceği shared expertler
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x:torch.Tensor)-> torch.Tensor:

        shape = x.size()
        x = x.view(-1, self.dim)
        # Gating mekanizması → skorlar (weights) ve seçilen uzmanların indeksleri
        weights, indices = self.gate(x)
        
        # Uzman çıktılarının toplanacağı boş tensor (aynı boyutta)
        y = torch.zeros_like(x)
        
        # Her uzmanın kaç kere seçildiğini sayılması. Bu MoE içinde bir experting çok fazla veya çok az kullanılmasını engellemek için
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        for i in range(self.expert_start_idx, self.expert_end_idx):
            expert = self.experts[i]

            if expert is None or counts[i] == 0:
                continue

            idx, top = torch.where(indices == i)
            #routed uzmanların çalıştırılması
            y = y.index_add(0, idx, expert(x[idx]) * weights[idx, top, None])
        # shared expertlerin çalıştırılması
        z = self.shared_experts(x)

        self.last_gate_scores = self.gate.last_scores
        self.last_gate_topk = self.gate.last_topk
        return (y + z + x).view(shape)


class Block(nn.Module):

    def __init__(self, layer_ids:int, args:ModelArgs):
        
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_ids < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:torch.Tensor)-> torch.Tensor:
        """
        Transformer bloğunun ileri (forward) geçiş işlemi.

        Parametreler (Args):
            x (torch.Tensor): Giriş tensörü.
            start_pos (int): Dizideki başlangıç pozisyonu (örneğin KV cache için).
            freqs_cis (torch.Tensor): Rotary positional embedding için önceden hesaplanmış karmaşık üstel (complex exponential) değerler.
            mask (Optional[torch.Tensor]): Belirli pozisyonları dikkat mekanizmasından hariç tutmak için kullanılan maske tensörü.

        Döndürür (Returns):
            torch.Tensor: Transformer bloğu işlemlerinden sonra elde edilen çıktı tensörü.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    """
    Pozisyonel gömmeler, çok katmanlı bloklar ve çıktı projeksiyonu içeren bir Transformer modeli.

    Öznitelikler (Attributes):
        max_seq_len (int): Transformer’ın desteklediği maksimum dizi (sequence) uzunluğu.
        embed (nn.Module): Giriş token'ları için gömme (embedding) katmanı.
        layers (torch.nn.ModuleList): Transformer bloklarını içeren katman listesi.
        norm (nn.Module): Tüm bloklardan sonra uygulanan katman normalizasyonu (LayerNorm).
        head (nn.Module): Modelin çıktısını kelime dağarcığına (vocab size) projekte eden son katman.
        freqs_cis (torch.Tensor): Rotary positional embedding için önceden hesaplanmış karmaşık üstel değerler (complex exponentials).
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """     
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, return_gate_info=False):

        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        moe_layer = None
        for l in reversed(self.layers):
            if isinstance(l.ffn, MoE):
                moe_layer = l.ffn
                break
        if return_gate_info:
            moe_layer = None
            for l in reversed(self.layers):
                if isinstance(l.ffn, MoE):
                    moe_layer = l.ffn
                    break
            return logits, moe_layer.last_gate_scores, moe_layer.last_gate_topk

        return logits

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
    print(model)
