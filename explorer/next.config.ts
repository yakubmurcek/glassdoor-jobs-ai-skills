import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: { unoptimized: true },
  async redirects() {
    return [
      { source: "/tiers", destination: "/analyze?group=country&metric=tier_mix", permanent: true },
      { source: "/job-families", destination: "/analyze?group=job_family&metric=ai_share", permanent: true },
    ];
  },
};

export default nextConfig;
