provider "google" {
  credentials = file("../../../gcp_api_key.json")
  project     = "epitechzoidberg"
  region      = "europe-west1"
}

locals {
  zones = [
    # "europe-west1-b",
    # "europe-west1-c",
    # "europe-west1-d",
    # "europe-central2-b",
    "europe-central2-c",
    # "europe-west4-a",
    # "europe-west4-b",
    # "europe-west4-c"
  ]
}

variable "USER_PASSWORD" {
  description = "The password for user"
  default     = "root"
}

variable "USER_NAME" {
  description = "The name of the user"
  default     = "root"
}

resource "google_compute_instance" "ai_training" {
  count        = 1
  name         = "ai-zoidberg-${count.index}"
  machine_type = "n1-highmem-2"
  zone         = local.zones[count.index]

  boot_disk {
    initialize_params {
      image = "family/debian-12"
      size  = 50
      type  = "pd-balanced"
    }
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  lifecycle {
    create_before_destroy = true
  }

  guest_accelerator {
    count = 2
    type  = "nvidia-tesla-t4"
  }

  network_interface {
    network = "default"

    access_config {
      // Ephemeral IP
    }
  }

  metadata = {
    ssh-keys       = "${var.USER_NAME}:${file("~/.ssh/id_ed25519.pub")}"
    startup-script = <<-EOF
        echo '${var.USER_NAME}:${var.USER_PASSWORD}' > /tmp/debug_password
        echo '${var.USER_NAME}:${var.USER_PASSWORD}' | chpasswd
        sed -i 's/^#\?PasswordAuthentication no/PasswordAuthentication yes/g' /etc/ssh/sshd_config
        systemctl restart ssh
    EOF
  }

}
