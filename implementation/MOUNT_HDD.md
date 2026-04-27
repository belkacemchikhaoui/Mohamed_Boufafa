# 🗂️ Mount the 1TB HDD (`/dev/sda2`) Before Running the Notebook

The preprocessing notebook writes to `/media/moamed/Data/yale-processed`.  
This requires the HDD (`/dev/sda2`, NTFS label `Data`) to be mounted first.

---

## ⚡ Quick Mount (Every Session)

Run this **once** each time you boot or plug in the drive:

```bash
sudo mkdir -p /media/moamed/Data
sudo mount -t ntfs-3g /dev/sda2 /media/moamed/Data
```

Verify it worked:

```bash
df -h /media/moamed/Data
ls /media/moamed/Data/yale-processed/
```

You should see free space and the `yale-processed/` folder.

---

## 🔁 Make It Mount Automatically on Boot (Optional but Recommended)

This mounts the HDD every time Linux starts — no manual command needed.

**Step 1 — Get the UUID:**
```bash
blkid /dev/sda2
```
Output will look like:
```
/dev/sda2: LABEL="Data" UUID="XXXX-XXXX" TYPE="ntfs"
```

**Step 2 — Add to `/etc/fstab`:**
```bash
sudo nano /etc/fstab
```

Add this line at the bottom (replace `XXXX-XXXX` with your actual UUID):
```
UUID=XXXX-XXXX  /media/moamed/Data  ntfs3  defaults,uid=1000,gid=1000,umask=022  0  0
```

> `ntfs3` is the modern Linux NTFS driver (kernel 5.15+). If it fails, use `ntfs-3g` instead.

**Step 3 — Test without rebooting:**
```bash
sudo mkdir -p /media/moamed/Data
sudo mount -a
df -h /media/moamed/Data
```

---

## 🩺 Troubleshoot

| Error | Fix |
|-------|-----|
| `Permission denied: '/media/moamed/Data'` | HDD is not mounted → run the Quick Mount command above |
| `mount: unknown filesystem type 'ntfs3'` | Use `ntfs-3g` instead: `sudo apt install ntfs-3g && sudo mount -t ntfs-3g /dev/sda2 /media/moamed/Data` |
| `mount: /dev/sda2 is already mounted` | Already mounted — check `df -h` |
| `Can't write to HDD` | Re-mount with write permissions: `sudo mount -t ntfs3 -o uid=1000,gid=1000 /dev/sda2 /media/moamed/Data` |

---

## 📋 After Mounting — Resume the Batch

Once the HDD is mounted, open the notebook and run these cells in order:

1. **Cell 3** (imports)
2. **Cell 5** (config + paths) — confirms `Output root: /media/moamed/Data/yale-processed (exists: True)`
3. **Cell 27** (pipeline function) — reloads `run_pipeline_visit` into kernel
4. **Cell 29** (batch) — resumes from where it left off (already-done visits are skipped)
