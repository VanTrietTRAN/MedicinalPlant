// Cấu hình hạ tầng Railway (Infrastructure as Code) — thay cho railway.json.
// Xem trước:  railway config plan     Áp dụng:  railway config apply
import { defineRailway, github, project, service } from "railway/iac";

// Repo này chỉ quản lý service "web" trong project.
export const partial = "web";

export default defineRailway(() => {
  const web = service("web", {
    source: github("VanTrietTRAN/MedicinalPlant", { branch: "main" }),
    build: {
      builder: "DOCKERFILE",
      dockerfilePath: "Dockerfile",
      // Chỉ build lại khi phần chạy app thay đổi (sửa research/ hay tài liệu thì không)
      watchPatterns: [
        "app.py",
        "medplant/**",
        "assets/**",
        "data/**",
        "examples/**",
        "weights/**",
        "scripts/**",
        "requirements.txt",
        "Dockerfile",
        ".railway/**",
      ],
    },
    // 1 replica tại Singapore — gần người dùng Việt Nam nhất
    replicas: { "asia-southeast1-eqsg3a": 1 },
    // Mặc định của Railway: restart ON_FAILURE, không ngủ (sleepApplication=false)
    deploy: {
      restartPolicyMaxRetries: 5,
    },
    healthcheck: "/",
    healthcheckTimeout: 300, // nạp 3 mô hình + warm-up (~45s trên Railway)
    env: {
      PORT: "8080", // cố định để khớp cổng đích của domain bên dưới
    },
    networking: {
      serviceDomains: {
        "web-production-68ba3.up.railway.app": { port: 8080 },
      },
    },
  });

  return project("medicinal-plant", {
    resources: [web],
  });
});
